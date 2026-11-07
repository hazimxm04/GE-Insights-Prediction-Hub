"""
chatbot/agent.py
================
The agent loop. The model sees the question and the tool list, and decides for itself which tool
to call, in what order, and when it has enough to answer. This replaces the old keyword router
(chatbot/chain/rag_chain.py), which decided the route in code.

    question -> [model: call tool? -> run tool -> result back to model] x N -> answer

Rules built into the loop (not left to the model):
  * at most MAX_STEPS model calls, so it cannot loop forever
  * a tool that fails returns an error to the model, which can try again; it never crashes the loop
  * every call is recorded in `steps`, so the page can show what the agent did and tests can check it
  * the client is passed in, so the loop is tested with a fake model and no API key

Tools: the data tools in chatbot/tools.py plus `what_if_seat` (live model call). A document search
tool can be added with `extra_tools` once the knowledge base is rebuilt.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chatbot import tools as data_tools                      # noqa: E402
from chatbot.chain.scenario_utils import pp_to_fraction      # noqa: E402

MAX_STEPS = 6
MAX_TOOL_CHARS = 6000          # a tool result longer than this is cut before the model sees it

SYSTEM_PROMPT = """You are the analyst assistant for GE-Insights (MYRamalan), a Malaysian state-assembly \
election predictor. "Harapan" is the target group; BN and PN are one group, "BN/PN".

Rules:
1. Every number, seat, probability or count in your answer must come from a tool result in this conversation. \
Never use numbers from memory. If you need a figure, call a tool.
2. There are two separate things: a FORECAST (a state that has not voted yet) and a VALIDATION (a state that \
has voted, the model scored against the real result). If a tool says something is unavailable, tell the user \
plainly that it is unavailable. Do not substitute the other one without saying so.
3. If a result has uses_fallback true, say the model had no reliable basis for that seat and used a simple \
base rate. Never state a confidence level; the tools do not provide one.
4. Sentiment is experimental: say so when you use it, and give the window dates.
5. Use as many tool calls as you need, but no more. If a question is outside what the tools cover, say so.
6. Keep the answer short and specific. Name the tool results you used (state, seat, forecast or validated)."""

WHAT_IF_SCHEMA = {
    "type": "function",
    "function": {
        "name": "what_if_seat",
        "description": ("Re-run the trained model for ONE named seat under a hypothetical change, e.g. 'what if "
                        "turnout were 10 points lower'. Use only for hypotheticals, not for actual results. "
                        "Changes are in PERCENTAGE POINTS (-10 = ten points lower)."),
        "parameters": {
            "type": "object",
            "properties": {
                "state": {"type": "string", "description": "State name, e.g. johor, neg_sembilan, melaka"},
                "seat": {"type": "string", "description": "Exact seat name, e.g. 'N.01 Buloh Kasap'"},
                "turnout_change_pp": {"type": "number", "description": "Turnout change in percentage points"},
                "margin_change_pp": {"type": "number", "description": "Winning-margin change in percentage points"},
            },
            "required": ["state", "seat"],
        },
    },
}


def what_if_seat(state: str, seat: str, turnout_change_pp=None, margin_change_pp=None) -> dict:
    """Live model call for one seat. Percentage points are converted to the model's fractions in code."""
    key = data_tools.normalise_state(state)
    if key is None:
        return data_tools._bad_state(state)
    try:
        turnout = pp_to_fraction(turnout_change_pp)
        margin = pp_to_fraction(margin_change_pp)
        from backend.core.models.state_predictor import StatePredictor     # lazy: needs the trained models
        predictor = StatePredictor(key)
        base = predictor.seat_baseline(seat)
        if base is None:
            return {"error": f"Seat '{seat}' not found in {data_tools.STATE_NAMES[key]}."}
        features = {k: base[k] for k in ('majority_change', 'turnout_change', 'incumbent_held',
                                         'log_voters', 'majority_perc_change', 'n_candidates_b')}
        features['incumbent_held'] = int(features['incumbent_held'])
        features['n_candidates_b'] = int(features['n_candidates_b'])
        if turnout is not None:
            features['turnout_change'] = turnout
        if margin is not None:
            features['majority_perc_change'] = margin
        r = predictor.predict_seat(seat, features)
        return {"state": data_tools.STATE_NAMES[key], "seat": seat,
                "scenario_side": "Harapan" if r['prediction'] == 'non-BN' else "BN/PN",
                "p_harapan": round(float(r['probability']), 3), "uses_fallback": bool(r['is_ood']),
                "baseline_turnout_change_pp": round(base['turnout_change'] * 100, 1),
                "scenario_turnout_change_pp": round(features['turnout_change'] * 100, 1)}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:                                                # missing model files, unknown seat
        return {"error": f"Could not run the model: {e}"}


def _registry(extra_tools=None):
    schemas = list(data_tools.TOOL_SCHEMAS) + [WHAT_IF_SCHEMA]
    funcs = dict(data_tools.TOOL_FUNCS)
    funcs["what_if_seat"] = what_if_seat
    for schema, func in (extra_tools or []):
        schemas.append(schema)
        funcs[schema["function"]["name"]] = func
    return schemas, funcs


def _run_tool(funcs: dict, name: str, arguments) -> dict:
    if name in data_tools.TOOL_FUNCS:
        return data_tools.call_tool(name, arguments)          # never raises, rejects bad arguments
    func = funcs.get(name)
    if func is None:
        return {"error": f"Unknown tool '{name}'."}
    try:
        args = json.loads(arguments) if isinstance(arguments, str) and arguments.strip() else (arguments or {})
        return func(**args)
    except TypeError as e:
        return {"error": f"Bad arguments for {name}: {e}"}
    except Exception as e:
        return {"error": f"{name} failed: {e}"}


def _to_message_text(result: dict) -> str:
    text = json.dumps(result, ensure_ascii=False, default=str)
    if len(text) > MAX_TOOL_CHARS:
        text = text[:MAX_TOOL_CHARS] + ' ...[cut: result too long, ask a narrower question]'
    return text


def _default_client():
    import os
    from groq import Groq
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY is not set")
    return Groq(api_key=key)


def run_agent(question: str, client=None, model: str = None, history=None, extra_tools=None,
              max_steps: int = MAX_STEPS) -> dict:
    """Answer one question. Returns {answer, steps, tools_used, stopped}.
    stopped is 'answered', 'step_limit' or 'error'. history is earlier {role, content} turns (text only)."""
    from sentiment.scoring.groq_settings import groq_model, completion_options
    client = client or _default_client()
    model = model or groq_model()
    schemas, funcs = _registry(extra_tools)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [m for m in (history or []) if m.get("role") in ("user", "assistant") and m.get("content")]
    messages.append({"role": "user", "content": question})

    steps = []
    for _ in range(max_steps):
        try:
            resp = client.chat.completions.create(model=model, messages=messages, tools=schemas,
                                                  tool_choice="auto", temperature=0.1,
                                                  **completion_options(model, 800))
        except Exception as e:
            return {"answer": f"The model call failed: {e}", "steps": steps,
                    "tools_used": [s["tool"] for s in steps], "stopped": "error"}
        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None) or []
        if not calls:
            text = (getattr(msg, "content", None) or "").strip()
            return {"answer": text or "The model returned an empty answer. Please ask again.",
                    "steps": steps, "tools_used": [s["tool"] for s in steps], "stopped": "answered"}
        messages.append({"role": "assistant", "content": getattr(msg, "content", None) or "",
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name,
                                                      "arguments": c.function.arguments or "{}"}} for c in calls]})
        for c in calls:
            result = _run_tool(funcs, c.function.name, c.function.arguments)
            steps.append({"tool": c.function.name, "arguments": c.function.arguments, "result": result})
            messages.append({"role": "tool", "tool_call_id": c.id, "content": _to_message_text(result)})

    return {"answer": "I could not finish within the step limit. Try a narrower question.",
            "steps": steps, "tools_used": [s["tool"] for s in steps], "stopped": "step_limit"}

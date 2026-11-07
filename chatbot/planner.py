"""
chatbot/planner.py
==================
Plan-then-act. Before the agent loop starts, one model call writes a short plan for the question:
which steps, and which tool each step needs. The plan is

  * shown to the user ("here is what I will do to answer this"),
  * given to the agent loop as guidance (it may still add or skip steps if a result says so),
  * ticked off as the tools are actually called, so the page can show progress and the evaluation can
    measure how much of the plan was followed.

The planner never calls a tool and never answers. If its output cannot be read, the plan is empty and
the agent simply runs without one. Reading, validating and tracking the plan is plain code (testable
without a model); only `make_plan` calls the model.
"""
import json
import re

MAX_PLAN_STEPS = 5
MAX_DO_CHARS = 160

PLANNER_PROMPT = """You plan how to answer a question about MYRamalan, a Malaysian state-election forecasting \
system. You do NOT answer it and you do NOT call tools. You write a short plan.

Tools available:
{catalogue}

Reply with ONLY a JSON object, no other text:
{{"steps": [{{"do": "<what this step finds out, plain words>", "tool": "<one tool name, or none>"}}],
  "clarify": ""}}

Rules:
- Use as few steps as the question needs: one step for a simple lookup, never more than {max_steps}.
- Every part of the question needs a step. A question that asks how the model did needs get_validation.
- Compare two states by one step per state.
- If the question names no state but needs one for a validation question, plan a step for each state that has \
voted (Johor and Negeri Sembilan) instead of asking.
- Put a question in "clarify" ONLY if the question cannot be answered at all without it; then "steps" is [].
- "how" and "why" questions about the system use search_docs.
- Numbers always come from tools, so never put numbers in the plan."""


def catalogue(schemas) -> str:
    lines = []
    for s in schemas:
        f = s["function"]
        first = re.split(r"(?<=[.!?])\s", f.get("description", "").strip())[0]
        lines.append(f"- {f['name']}: {first[:170]}")
    return "\n".join(lines)


def parse_plan(text: str, known_tools) -> dict:
    """Plan from the model's reply. {'steps': [{'do','tool'}], 'clarify': str, 'ok': bool}. Never raises.
    Unknown tool names become 'none'; steps are capped; anything unreadable gives an empty plan."""
    empty = {"steps": [], "clarify": "", "ok": False}
    try:
        m = re.search(r"\{.*\}", text or "", re.S)
        data = json.loads(m.group(0)) if m else None
    except (ValueError, TypeError):
        return empty
    if not isinstance(data, dict):
        return empty
    known, steps = set(known_tools), []
    for raw in (data.get("steps") or [])[:MAX_PLAN_STEPS]:
        if not isinstance(raw, dict):
            continue
        do = str(raw.get("do") or "").strip()[:MAX_DO_CHARS]
        tool = str(raw.get("tool") or "none").strip()
        if do:
            steps.append({"do": do, "tool": tool if tool in known else "none"})
    clarify = str(data.get("clarify") or "").strip()[:300] if not steps else ""
    return {"steps": steps, "clarify": clarify, "ok": bool(steps or clarify)}


def make_plan(question, schemas, client, model, history=None) -> dict:
    """One model call, no tools. Returns the parsed plan; an empty plan if the call or the parsing fails."""
    from sentiment.scoring.groq_settings import completion_options
    known = [s["function"]["name"] for s in schemas]
    system = PLANNER_PROMPT.format(catalogue=catalogue(schemas), max_steps=MAX_PLAN_STEPS)
    turns = [m for m in (history or [])[-4:] if m.get("role") in ("user", "assistant") and m.get("content")]
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.1, **completion_options(model, 500),
            messages=[{"role": "system", "content": system}, *turns, {"role": "user", "content": question}])
        return parse_plan(resp.choices[0].message.content or "", known)
    except Exception:
        return {"steps": [], "clarify": "", "ok": False}


def plan_text(plan: dict) -> str:
    """The plan as guidance for the agent loop, or '' for an empty plan."""
    if not plan or not plan.get("steps"):
        return ""
    lines = [f"{i}. {s['do']}" + (f" (tool: {s['tool']})" if s["tool"] != "none" else "")
             for i, s in enumerate(plan["steps"], 1)]
    return ("Plan for this question. Follow it, and add a step if a result shows you need one:\n" + "\n".join(lines))


def mark_progress(plan: dict, tool_calls) -> list:
    """Plan steps with a 'done' flag. Each tool call ticks the first not-yet-done step that names that tool;
    a step with tool 'none' is done once every tool step before it is done and the run has answered
    (the caller passes answered=True through finish_plan)."""
    steps = [{**s, "done": False} for s in (plan or {}).get("steps", [])]
    for tool in tool_calls:
        for s in steps:
            if not s["done"] and s["tool"] == tool:
                s["done"] = True
                break
    return steps


def finish_plan(plan: dict, tool_calls, answered: bool) -> list:
    """Final plan state: tool steps ticked by calls made; 'none' steps (thinking/summarising) ticked if answered."""
    steps = mark_progress(plan, tool_calls)
    if answered:
        for s in steps:
            if s["tool"] == "none":
                s["done"] = True
    return steps


def followed(plan_steps: list) -> float:
    """Share of tool steps that were actually called (1.0 when the plan has no tool steps)."""
    tool_steps = [s for s in plan_steps if s["tool"] != "none"]
    return 1.0 if not tool_steps else sum(s["done"] for s in tool_steps) / len(tool_steps)

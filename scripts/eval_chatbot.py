"""
scripts/eval_chatbot.py
=======================
Runs the question set (eval/chatbot_questions.csv) through up to four chatbot set-ups and scores each answer.

  old             the original keyword-routed RAG chatbot (chatbot/chain/rag_chain.py)
  router          code picks ONE tool by keywords, the model only writes the answer from it
  agent           the model picks tools in a loop (chatbot/agent.py)
  agent_verified  the agent plus the number check (chatbot/verifier.py)
  agent_planned   the agent plus a written plan first (chatbot/planner.py); also records how much of the plan it followed

Scoring is automatic and deliberately plain:
  content_pass   every word in must_contain appears, at least one of must_contain_any appears,
                 none of must_not_contain appears (case-insensitive; numbers match whole numbers)
  tool_pass      every expected tool was called (agent modes and router only; the old chatbot has no tools)
  pass           content_pass and tool_pass
  cause          why a question failed: error | wrong_tool | missing_content | forbidden_phrase | unsupported_number
Words are a crude judge of an answer. Read the failures, not just the totals.

Progress is saved after every question, so the Groq daily limit never loses work. Run again with the
same command and it carries on (`--fresh` starts over).

  python scripts/eval_chatbot.py --modes agent agent_verified
  python scripts/eval_chatbot.py --modes old router --skip-kb
  python scripts/eval_chatbot.py --limit 5            # try the first 5 questions
"""
import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUESTIONS = ROOT / "eval" / "chatbot_questions.csv"
RESULTS = ROOT / "eval" / "chatbot_eval_results.csv"
SUMMARY = ROOT / "eval" / "chatbot_eval_summary.md"
ALL_MODES = ["old", "router", "agent", "agent_verified", "agent_planned"]
COLS = ["id", "mode", "category", "answer", "tools_used", "content_pass", "tool_pass", "pass", "cause",
        "unsupported", "plan_followed", "seconds"]
RATE_LIMIT_WORDS = ("rate limit", "429", "rate_limit", "tokens per day", "tpd")


# ── scoring (pure) ──────────────────────────────────────────────────────────────

def _split(cell) -> list:
    return [t.strip() for t in str(cell or "").split("|") if t.strip()]


def contains(text: str, token: str) -> bool:
    """Case-insensitive; a purely numeric token must match a whole number (6 does not match 16 or 0.6)."""
    if re.fullmatch(r"\d+(\.\d+)?", token):
        return re.search(rf"(?<![\d.]){re.escape(token)}(?![\d]|\.\d)", text) is not None
    return token.lower() in text.lower()


def score_content(answer: str, must_all, must_any, must_not) -> tuple:
    """(passed, cause) for the words in an answer."""
    all_, any_, no_ = _split(must_all), _split(must_any), _split(must_not)
    if any(contains(answer, t) for t in no_):
        return False, "forbidden_phrase"
    if any(not contains(answer, t) for t in all_) or (any_ and not any(contains(answer, t) for t in any_)):
        return False, "missing_content"
    return True, ""


def score_tools(expected, used) -> bool:
    return all(t in set(used or []) for t in _split(expected))


def is_rate_limit(text: str) -> bool:
    return any(w in (text or "").lower() for w in RATE_LIMIT_WORDS)


# ── the four set-ups ────────────────────────────────────────────────────────────

ROUTES = [                                   # (keywords, tool); first match wins; the code decides, not the model
    (("what if", "suppose", "hypothetical"), "what_if_seat"),
    (("sentiment", "news", "headline"), "get_sentiment"),
    (("flip", "changed side", "change side", "change hands", "switch"), "list_flips"),
    (("wrong", "accurate", "accuracy", "baseline", "beat", "did the model do", "right"), "get_validation"),
    (("forecast", "predicted outcome", "predict", "will happen"), "get_state_forecast"),
    (("why ", "how does", "how is", "what is a", "how do"), "search_docs"),
]


def route(question: str):
    q = question.lower()
    for words, tool in ROUTES:
        if any(w in q for w in words):
            return tool
    return None


def run_router(question: str, client=None, model=None) -> dict:
    """Fixed pipeline: keywords pick one tool and its state, the model writes the answer from the result."""
    from chatbot import agent, tools as T
    from chatbot.docs_search import search_docs
    from chatbot.chain.scenario_utils import detect_state
    from sentiment.scoring.groq_settings import groq_model, completion_options
    client, model = client or agent._default_client(), model or groq_model()
    tool, state, steps = route(question), detect_state(question), []
    if tool == "search_docs":
        steps.append({"tool": tool, "arguments": "{}", "result": search_docs(question)})
    elif tool in ("get_sentiment",):
        steps.append({"tool": tool, "arguments": "{}", "result": T.get_sentiment()})
    elif tool and state:
        steps.append({"tool": tool, "arguments": "{}", "result": T.call_tool(tool, {"state": state})})
    try:
        evidence = agent._to_message_text(steps[0]["result"]) if steps else "No tool was run."
        resp = client.chat.completions.create(
            model=model, temperature=0.1, **completion_options(model, 600),
            messages=[{"role": "system", "content": agent.SYSTEM_PROMPT},
                      {"role": "user", "content": f"Tool result:\n{evidence}\n\nQuestion: {question}"}])
        answer = (resp.choices[0].message.content or "").strip() or "The model returned an empty answer."
        return {"answer": answer, "steps": steps, "tools_used": [s["tool"] for s in steps], "stopped": "answered"}
    except Exception as e:
        return {"answer": f"The model call failed: {e}", "steps": steps, "tools_used": [], "stopped": "error"}


def run_old(question: str, _state={}) -> dict:
    from chatbot.chain.rag_chain import ask, get_collection
    if "collection" not in _state:
        _state["collection"] = get_collection()
    try:
        r = ask(question, _state["collection"])
        return {"answer": r.get("answer", ""), "steps": [], "tools_used": [], "stopped": "answered"}
    except Exception as e:
        return {"answer": f"The old chatbot failed: {e}", "steps": [], "tools_used": [], "stopped": "error"}


def run_mode(mode: str, question: str, client=None, model=None) -> dict:
    from chatbot import agent
    if mode == "old":
        return run_old(question)
    if mode == "router":
        return run_router(question, client, model)
    return agent.run_agent(question, client=client, model=model, verify=(mode == "agent_verified"),
                           plan=(mode == "agent_planned"))


# ── one row ─────────────────────────────────────────────────────────────────────

def evaluate_one(q: dict, mode: str, runner=run_mode) -> dict:
    from chatbot.verifier import check_answer
    t0 = time.time()
    r = runner(mode, q["question"])
    answer, used = r.get("answer", ""), r.get("tools_used", [])
    content_ok, cause = score_content(answer, q.get("must_contain"), q.get("must_contain_any"), q.get("must_not_contain"))
    tool_ok = True if mode == "old" else score_tools(q.get("expected_tools"), used)
    unsupported = []
    if mode != "old":                                   # the old chatbot has no tool results to check against
        unsupported = check_answer(answer.split("\n\n⚠️")[0], r.get("steps", []), q["question"])["unsupported"]
    if r.get("stopped") == "error":
        cause = "error"
    elif not tool_ok:
        cause = "wrong_tool"
    elif not content_ok:
        pass
    elif unsupported:
        cause = "unsupported_number"
    passed = r.get("stopped") != "error" and content_ok and tool_ok and not unsupported
    return {"id": q["id"], "mode": mode, "category": q.get("category", ""), "answer": answer,
            "tools_used": ",".join(used), "content_pass": content_ok, "tool_pass": tool_ok, "pass": passed,
            "cause": "" if passed else cause, "unsupported": ",".join(unsupported),
            "plan_followed": r.get("plan_followed", ""),
            "seconds": round(time.time() - t0, 1)}


# ── summary ─────────────────────────────────────────────────────────────────────

def summarise(res: pd.DataFrame) -> str:
    res = res.copy()
    res["pass"] = res["pass"].astype(bool)
    lines = ["# Chatbot evaluation", "", "| set-up | questions | passed | pass rate |", "|---|---|---|---|"]
    for mode in [m for m in ALL_MODES if m in set(res["mode"])]:
        g = res[res["mode"] == mode]
        lines.append(f"| {mode} | {len(g)} | {int(g['pass'].sum())} | {g['pass'].mean():.0%} |")
    if "plan_followed" in res and (res["mode"] == "agent_planned").any():
        pf = pd.to_numeric(res.loc[res["mode"] == "agent_planned", "plan_followed"], errors="coerce").dropna()
        if len(pf):
            lines += ["", f"Plan followed (agent_planned): on average {pf.mean():.0%} of the planned tool steps were called."]
    fails = res[~res["pass"]]
    if len(fails):
        lines += ["", "## Why answers failed", "", "| set-up | cause | count |", "|---|---|---|"]
        for (mode, cause), n in fails.groupby(["mode", "cause"]).size().items():
            lines.append(f"| {mode} | {cause or 'unknown'} | {n} |")
    cats = res.groupby(["category", "mode"])["pass"].agg(["sum", "count"]).reset_index()
    lines += ["", "## By question type (passed / asked)", ""]
    modes = [m for m in ALL_MODES if m in set(res["mode"])]
    lines += ["| type | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    for cat in sorted(set(cats["category"])):
        row = []
        for m in modes:
            c = cats[(cats["category"] == cat) & (cats["mode"] == m)]
            row.append(f"{int(c['sum'].iloc[0])}/{int(c['count'].iloc[0])}" if len(c) else "-")
        lines.append(f"| {cat} | " + " | ".join(row) + " |")
    lines += ["", "Pass = the right words appear, the expected tools were called, and (for tool-based set-ups) every number "
                  "in the answer is supported by a tool result. Wording checks are crude; read the failed answers in "
                  "`eval/chatbot_eval_results.csv`.", ""]
    return "\n".join(lines)


# ── main ────────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["agent", "agent_verified"], choices=ALL_MODES)
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions")
    ap.add_argument("--skip-kb", action="store_true", help="skip questions that need the documents (requires_kb = yes)")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between questions")
    ap.add_argument("--fresh", action="store_true", help="ignore saved results and start over")
    args = ap.parse_args(argv)

    qs = pd.read_csv(QUESTIONS, dtype=str).fillna("")
    if args.skip_kb:
        qs = qs[qs["requires_kb"].str.lower() != "yes"]
    if args.limit:
        qs = qs.head(args.limit)
    done = pd.DataFrame(columns=COLS) if args.fresh or not RESULTS.exists() else pd.read_csv(RESULTS, dtype=str)
    seen = set(zip(done["id"], done["mode"]))
    rows = done.to_dict("records")

    for mode in args.modes:
        for q in qs.to_dict("records"):
            if (q["id"], mode) in seen:
                continue
            row = evaluate_one(q, mode)
            print(f"{q['id']:>4} {mode:<15} {'PASS' if row['pass'] else 'FAIL ' + row['cause']}")
            if row["cause"] == "error" and is_rate_limit(row["answer"]):
                print("\nGroq limit reached. Progress is saved; run the same command later to carry on.")
                pd.DataFrame(rows, columns=COLS).to_csv(RESULTS, index=False)
                return 2
            rows.append(row)
            pd.DataFrame(rows, columns=COLS).to_csv(RESULTS, index=False)
            time.sleep(args.sleep)

    res = pd.DataFrame(rows, columns=COLS)
    text = summarise(res)
    SUMMARY.write_text(text, encoding="utf-8")
    print("\n" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
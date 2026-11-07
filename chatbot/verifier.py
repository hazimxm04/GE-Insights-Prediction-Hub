"""
chatbot/verifier.py
===================
Checks an agent answer against the tool results it was built from. Pure code, no model call.

A number in the answer is SUPPORTED if it appears in a tool result, in the question, or can be
derived from tool numbers in a simple way (a share written as a percentage, a difference or a sum
of two tool numbers, such as "2 seats behind" from 48 and 50). Anything else is UNSUPPORTED:
the model produced it from nowhere, which is the failure the agent exists to prevent.

This checks numbers only. It does not check that the words are right, or that the right tool was used.
"""
import json
import re
from itertools import combinations

NUM_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?%?")
SEAT_ID_RE = re.compile(r"\bN\.?\s?\d{1,2}\b", re.I)          # N.12 is a name, not a figure
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
LIST_MARKER_RE = re.compile(r"(?m)^\s*\d+[.)]\s")             # "1. " at the start of a line
TOL = 0.051                                                    # one decimal place of rounding


def _numbers(text: str) -> list:
    """Numbers in a text as floats; percentages are returned as the percentage number (85.7% -> 85.7)."""
    text = DATE_RE.sub(" ", SEAT_ID_RE.sub(" ", LIST_MARKER_RE.sub("", text or "")))
    out = []
    for tok in NUM_RE.findall(text):
        try:
            out.append(float(tok.rstrip("%")))
        except ValueError:
            pass
    return out


def _walk(obj, acc: list):
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        acc.append(float(obj))
    elif isinstance(obj, str):
        acc.extend(_numbers(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk(v, acc)


def _supported_values(question: str, steps: list) -> set:
    base = []
    for s in steps:
        _walk(s.get("result"), base)
        _walk(s.get("arguments") if not isinstance(s.get("arguments"), str) else _safe_json(s["arguments"]), base)
    base += _numbers(question)
    vals = set(round(v, 4) for v in base)
    for v in list(vals):                                       # 0.857 can be written 85.7%
        if -1.0 <= v <= 1.0:
            vals.add(round(v * 100, 4))
    ints = sorted({v for v in vals if float(v).is_integer() and abs(v) <= 500})
    for a, b in combinations(ints, 2):                         # "2 seats behind", "total of 14"
        vals.add(abs(a - b))
        vals.add(a + b)
    return vals


def _safe_json(s):
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return None


def check_answer(answer: str, steps: list, question: str = "") -> dict:
    """{'ok': bool, 'unsupported': [numbers as written], 'checked': n}.
    An answer with no numbers is ok. Years are checked like any other number (the model once wrote 2021 for
    Johor's previous election, which was 2022), except 2026, the election cycle the project is about."""
    vals = _supported_values(question, steps or [])
    bad, checked = [], 0
    for tok in NUM_RE.findall(DATE_RE.sub(" ", SEAT_ID_RE.sub(" ", LIST_MARKER_RE.sub("", answer or "")))):
        try:
            x = float(tok.rstrip("%"))
        except ValueError:
            continue
        if tok == "2026":                                  # the election cycle this whole project is about
            continue
        checked += 1
        if not any(abs(x - v) <= TOL for v in vals):
            bad.append(tok)
    return {"ok": not bad, "unsupported": sorted(set(bad), key=bad.index), "checked": checked}


def retry_message(unsupported: list) -> str:
    return ("Your answer contains numbers that no tool result supports: " + ", ".join(unsupported) +
            ". Rewrite the answer using only numbers from the tool results above. If you need another figure, "
            "call a tool. If it is not available, say so.")
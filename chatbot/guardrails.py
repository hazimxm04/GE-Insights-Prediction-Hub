"""
chatbot/guardrails.py
=====================
Checks that run in CODE, around the model, so they do not depend on the model following a prompt.

  check_input    before the model sees a question: too long, prompt-injection wording, off topic, greeting
  scrub          on tool results before the model reads them: injection wording inside data is removed
  check_output   on the finished answer: leaked system prompt or key-like strings are replaced

What this is and is not. These are keyword and pattern rules. They stop the obvious cases, cost no model
call, and can be tested. They do NOT make the agent injection-proof: a cleverly worded request can get
past a keyword list, and an unusual but legitimate question can be blocked. The tools are read-only, so
the worst a successful injection can do today is make the agent say something off topic.
"""
import re

MAX_QUESTION_CHARS = 800
MAX_STRING_IN_RESULT = 400

REFUSAL_OFF_TOPIC = ("I can only help with MYRamalan: Malaysian state-assembly election forecasts, how the model "
                     "performed against real results, news sentiment, and how the system works. "
                     "Try asking about a state or a seat.")
REFUSAL_INJECTION = ("I can't follow instructions that change my role or ask for my configuration. "
                     "I can answer questions about the election forecasts, the validation results, sentiment and "
                     "how the system works.")
REFUSAL_TOO_LONG = f"That question is too long. Please keep it under {MAX_QUESTION_CHARS} characters."
GREETING_REPLY = ("Hi! I answer questions about MYRamalan's Malaysian state-election forecasts: for example how the "
                  "model did in Johor, which seats flipped in Negeri Sembilan, or the forecast for Melaka.")
EMPTY_REPLY = "Ask me something about the election forecasts, the validation results, or sentiment."
OUTPUT_REFUSAL = "I can't share that. I can answer questions about the election forecasts and how the system works."

INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(instruction|prompt|rule|guideline|polic|restriction)",
    r"\b(ignore|disregard|forget)\b\s+(all\s+|any\s+|the\s+|your\s+)*(previous|prior|above|earlier)\b(?!\s+election)",
    r"\b(system|developer|hidden|initial)\s+(prompt|message|instruction)",
    r"\b(reveal|show|print|repeat|display|leak|tell me)\b.{0,30}\b(your|the)\b.{0,20}\b(instruction|prompt|rules|configuration|config|system)",
    r"\byou are now\b", r"\bfrom now on you\b", r"\bact as\b", r"\bpretend (to be|you)\b", r"\brole[- ]?play\b",
    r"\b(jailbreak|DAN mode|developer mode)\b", r"\bnew instructions?\b",
    r"\b(api[ _-]?key|secret|password|token)s?\b.{0,20}\b(show|give|print|reveal|what)\b",
    r"\b(show|give|print|reveal|what('?s| is))\b.{0,20}\b(api[ _-]?key|secret key|password)\b",
    r"\b(abaikan|lupakan|langkau)\b.{0,30}\b(arahan|peraturan)", r"\barahan (sistem|asal)\b", r"\bberpura-?pura\b",
]
_INJECTION = [re.compile(p, re.I | re.S) for p in INJECTION_PATTERNS]

TOPIC_WORDS = [
    # places
    "johor", "negeri sembilan", "sembilan", "selangor", "melaka", "malacca", "perak", "malaysia", "state",
    # elections and parties
    "election", "pilihan raya", "prn", "pru", "seat", "kerusi", "dun", "constituency", "vote", "voter", "undi",
    "harapan", "pakatan", "ph", "bn", "pn", "barisan", "perikatan", "umno", "pas", "bersatu", "dap", "pkr",
    "amanah", "ns", "opposition", "coalition", "anwar", "muhyiddin", "zahid", "politic", "politik", "government",
    "kerajaan",
    # what the system does
    "forecast", "predict", "ramalan", "model", "accura", "validat", "baseline", "flip", "fallback", "probab",
    "sentiment", "headline", "turnout", "majority", "confiden", "ood", "demograph", "ethnic", "feature", "leakage",
    "limitation", "frozen", "myramalan", "ge-insights", "trained", "training", "what if", "scenario", "last time",
    "previous election", "persistence", "base rate", "auc", "news", "scoring", "agent",
]
_EXACT = {"ph", "bn", "pn", "pas", "dap", "pkr", "ood", "auc", "dun", "prn", "pru", "ns"}
_SHORT = [w for w in TOPIC_WORDS if w in _EXACT]        # whole words only: 'ph' must not match 'photo'
_LONG = [w for w in TOPIC_WORDS if w not in _EXACT]     # word starts: 'seat' also matches 'seats'
_TOPIC = re.compile(r"\b(" + "|".join(re.escape(w) for w in _SHORT) + r")\b|(?<![a-z])(" +
                    "|".join(re.escape(w) for w in sorted(_LONG, key=len, reverse=True)) + ")", re.I)
_SEAT = re.compile(r"\bN\.?\s?\d{1,2}\b", re.I)
_GREETING = re.compile(r"^\s*(hi|hello|hey|yo|salam|assalamualaikum|good (morning|afternoon|evening)|thanks?|"
                       r"thank you|terima kasih|ok|okay)\W*$", re.I)
_FOLLOW_UP_WORDS = 8              # a short question inside a conversation that is already on topic is allowed


def check_input(question, history=None) -> dict:
    """{'allowed': bool, 'reason': None|'empty'|'too_long'|'injection'|'off_topic'|'greeting', 'message': str}.
    message is the fixed reply to show when not allowed."""
    text = (question or "").strip()
    if not text:
        return {"allowed": False, "reason": "empty", "message": EMPTY_REPLY}
    if len(text) > MAX_QUESTION_CHARS:
        return {"allowed": False, "reason": "too_long", "message": REFUSAL_TOO_LONG}
    if _GREETING.match(text):
        return {"allowed": False, "reason": "greeting", "message": GREETING_REPLY}
    if any(p.search(text) for p in _INJECTION):
        return {"allowed": False, "reason": "injection", "message": REFUSAL_INJECTION}
    on_topic = bool(_TOPIC.search(text) or _SEAT.search(text))
    in_conversation = bool(history) and len(text.split()) <= _FOLLOW_UP_WORDS
    if not (on_topic or in_conversation):
        return {"allowed": False, "reason": "off_topic", "message": REFUSAL_OFF_TOPIC}
    return {"allowed": True, "reason": None, "message": ""}


def scrub(obj):
    """Tool results are DATA. Remove injection wording from any text in them (a scraped headline could hold some)
    and cut very long strings. Returns a copy."""
    if isinstance(obj, str):
        if any(p.search(obj) for p in _INJECTION):
            return "[text removed: looked like an instruction]"
        return obj if len(obj) <= MAX_STRING_IN_RESULT else obj[:MAX_STRING_IN_RESULT] + "..."
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub(v) for v in obj]
    return obj


_KEY_LIKE = re.compile(r"\b(gsk_[A-Za-z0-9]{16,}|sk-[A-Za-z0-9]{20,}|[A-Za-z0-9_\-]{32,}\.[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{20,})\b")


def _norm(text: str) -> list:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def leaks_prompt(answer: str, system_prompt: str, window: int = 20) -> bool:
    """True if the answer repeats `window` words in a row from the system prompt."""
    a, s = _norm(answer), _norm(system_prompt)
    shingles = {" ".join(s[i:i + window]) for i in range(max(0, len(s) - window + 1))}
    return any(" ".join(a[i:i + window]) in shingles for i in range(max(0, len(a) - window + 1)))


def check_output(answer: str, system_prompt: str = "") -> dict:
    """{'ok': bool, 'reason': None|'key'|'prompt_leak', 'answer': safe text}."""
    if _KEY_LIKE.search(answer or ""):
        return {"ok": False, "reason": "key", "answer": OUTPUT_REFUSAL}
    if system_prompt and leaks_prompt(answer, system_prompt):
        return {"ok": False, "reason": "prompt_leak", "answer": OUTPUT_REFUSAL}
    return {"ok": True, "reason": None, "answer": answer}

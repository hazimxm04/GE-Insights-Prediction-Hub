"""
chatbot/chain/scenario_utils.py
===============================
Small pure helpers for the chatbot. No Groq, no ChromaDB, so they can be tested
without an API key.

Why a separate file: the model's turnout feature is a FRACTION (0.10 = 10 points),
but people (and the LLM) say "10%". The conversion lives here, in code, so it
cannot depend on the LLM remembering the right scale.
"""
import re

# a "what if" must change something about a seat; plain "if the ..." is too broad
SCENARIO_KEYWORDS = [
    'what if', 'if turnout', 'if the turnout', 'if the majority', 'if the margin',
    'suppose', 'hypothetically', 'what would happen',
]

MAX_ABS_PP = 50.0      # a swing bigger than 50 points is a typo, not a scenario


def is_scenario_question(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in SCENARIO_KEYWORDS)


def pp_to_fraction(value_pp):
    """Percentage points -> fraction (-10 -> -0.10). None stays None.
    Raises ValueError for swings the model has never seen (|x| > 50 points)."""
    if value_pp is None:
        return None
    value_pp = float(value_pp)
    if abs(value_pp) > MAX_ABS_PP:
        raise ValueError(f'{value_pp} percentage points is outside the believable range '
                         f'(+/-{MAX_ABS_PP:.0f}). Did you mean a smaller change?')
    return value_pp / 100.0


def detect_state(question: str):
    """State named in a question, or None. Uses whole words, so 'predictions' is not 'NS'."""
    q = question.lower()
    if re.search(r'\bns\b', q) or 'negeri sembilan' in q or 'sembilan' in q:
        return 'neg_sembilan'
    if re.search(r'\b(johor|jhr)\b', q):
        return 'johor'
    if re.search(r'\b(melaka|malacca)\b', q):
        return 'melaka'
    if 'selangor' in q:
        return 'selangor'
    if 'perak' in q:
        return 'perak'
    return None


def message_text(msg, fallback: str = '') -> str:
    """Text of an LLM message; empty or missing content (e.g. a reasoning model that ran
    out of tokens) gives the fallback instead of crashing."""
    content = getattr(msg, 'content', None)
    text = (content or '').strip()
    return text if text else fallback

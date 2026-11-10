"""
chatbot/docs_search.py
======================
The agent's `search_docs` tool: finds passages in the plain-text explanations under
chatbot/knowledge_base/docs/ (how the model works, fallback seats, baselines, sentiment, limits).

These documents carry NO state figures. Numbers come from the data tools in chatbot/tools.py,
so a number can never be stale in a document.

Search is word matching with a small stemmer, not embeddings. The corpus is a few dozen short
passages, so this is enough, needs no vector database or model download, and works on any host
with nothing to build at deploy time. Passages are the "## " sections of each markdown file.
"""
import json
import re
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent / "knowledge_base" / "docs"
MAX_CHARS = 1200               # one passage is cut to this before the model sees it

_STOP = set("a an and are as at be by do does for from how i in is it its of on or that the this to was what when "
            "where which who why with you your me my we our can will not no".split())

_chunks_cache = None


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _tokens(text: str) -> list:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1]


def load_chunks(docs_dir=None) -> list:
    """[{source, title, section, text}] with one chunk per '## ' section. Skips files that cannot be read."""
    out = []
    for path in sorted(Path(docs_dir or DOCS_DIR).glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        title, section, buf = path.stem, "", []

        def flush():
            body = "\n".join(buf).strip()
            if body:
                out.append({"source": path.name, "title": title, "section": section, "text": body})

        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("## "):
                flush()
                section, buf = line[3:].strip(), []
            else:
                buf.append(line)
        flush()
    return out


def _chunks():
    global _chunks_cache
    if _chunks_cache is None:
        _chunks_cache = load_chunks()
    return _chunks_cache


def search_docs(query: str, k: int = 3) -> dict:
    """Best passages for a question about HOW the system works. Never raises."""
    try:
        k = max(1, min(int(k), 5))
    except (TypeError, ValueError):
        k = 3
    q = set(_tokens(str(query or "")))
    if not q:
        return {"query": query, "results": [], "note": "Give a few keywords about what you want to know."}
    scored = []
    for c in _chunks():
        body = _tokens(c["text"])
        head = set(_tokens(c["title"] + " " + c["section"]))
        if not body:
            continue
        hits = sum(1 for w in q if w in body or w in head)
        score = hits + 1.5 * len(q & head) + 0.02 * sum(body.count(w) for w in q)
        if hits:
            scored.append((score, c))
    scored.sort(key=lambda s: -s[0])
    results = [{"source": c["source"], "section": f'{c["title"]} / {c["section"]}'.strip(" /"),
                "text": c["text"][:MAX_CHARS]} for _, c in scored[:k]]
    note = "" if results else "No passage matched. Try different keywords, or say the documents do not cover it."
    return {"query": query, "results": results, "note": note or
            "Explanations only. Take every figure from the data tools, not from these passages."}


SEARCH_DOCS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": ("Search the written explanations of HOW the system works: the model and its features, fallback "
                        "seats, why there is no confidence level, baselines and how the model is judged, the "
                        "experimental sentiment layer, limitations, and terms (DUN, OOD, flip). Use it for 'how' and "
                        "'why' questions. It holds NO figures: get numbers from the data tools."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A few keywords, e.g. 'fallback base rate'"},
                "k": {"type": "integer", "description": "How many passages (1 to 5, default 3)"},
            },
            "required": ["query"],
        },
    },
}

"""
Which Groq model the sentiment scorer calls, and how to call it.

Groq retires models from time to time (llama-3.1-8b-instant was shut down on 16 Aug 2026 and
now returns 404 model_not_found). The name therefore lives here, can be overridden with the
GROQ_MODEL environment variable, and is not hard-coded in the scorer.

Reasoning models such as openai/gpt-oss-20b spend part of max_tokens on thinking before they
answer. With the old limit of 100 the answer can be cut off, so for those models the limit is
raised and the reasoning effort is set to "low".

This file imports nothing from groq, so it can be tested without the package or an API key.
"""
import os

DEFAULT_MODEL = "openai/gpt-oss-20b"      # Groq's recommended replacement for llama-3.1-8b-instant
MIN_TOKENS_REASONING = 1500


def groq_model() -> str:
    """The model name to use: GROQ_MODEL if set, otherwise the default."""
    return os.getenv("GROQ_MODEL", "").strip() or DEFAULT_MODEL


def completion_options(model: str, max_tokens: int) -> dict:
    """Extra arguments for client.chat.completions.create(...) for this model."""
    if "gpt-oss" in model:
        return {"max_tokens": max(max_tokens, MIN_TOKENS_REASONING), "reasoning_effort": "low"}
    return {"max_tokens": max_tokens}

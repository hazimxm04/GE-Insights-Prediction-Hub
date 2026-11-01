"""
The sentiment scorer must not be hard-wired to a retired Groq model, and reasoning models
must be given enough tokens to finish their answer (a 100-token limit can cut it off).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from sentiment.scoring import groq_settings as gs


def test_default_is_not_the_retired_model(monkeypatch):
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    assert gs.groq_model() == gs.DEFAULT_MODEL
    assert gs.DEFAULT_MODEL != "llama-3.1-8b-instant"


def test_environment_variable_overrides_the_model(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "some/other-model")
    assert gs.groq_model() == "some/other-model"
    monkeypatch.setenv("GROQ_MODEL", "   ")          # blank falls back to the default
    assert gs.groq_model() == gs.DEFAULT_MODEL


def test_reasoning_model_gets_room_to_answer():
    opts = gs.completion_options("openai/gpt-oss-20b", 100)
    assert opts["max_tokens"] >= gs.MIN_TOKENS_REASONING
    assert opts["reasoning_effort"] == "low"


def test_other_models_keep_the_requested_limit():
    assert gs.completion_options("llama-3.3-70b-versatile", 100) == {"max_tokens": 100}


def test_larger_limit_is_not_reduced():
    assert gs.completion_options("openai/gpt-oss-20b", 4000)["max_tokens"] == 4000


def test_scorer_uses_the_settings_not_a_hard_coded_model():
    src = (ROOT / "sentiment" / "scoring" / "sentiment_scorer.py").read_text(encoding="utf-8")
    assert 'MODEL  = "llama' not in src and "llama-3.1-8b-instant\"" not in src
    assert "groq_model()" in src and "completion_options(" in src

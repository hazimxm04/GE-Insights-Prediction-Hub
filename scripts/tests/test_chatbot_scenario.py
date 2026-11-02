"""
Chatbot helpers (no Groq key needed).

The bug these guard against: the model's turnout feature is a fraction (-0.10), but
the LLM was told to send percentage points (-10), which is 100x too big.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from chatbot.chain.scenario_utils import (
    is_scenario_question, pp_to_fraction, detect_state, message_text,
)


def test_percentage_points_become_fractions():
    assert pp_to_fraction(-10) == pytest.approx(-0.10)
    assert pp_to_fraction(5) == pytest.approx(0.05)
    assert pp_to_fraction(None) is None


def test_absurd_swings_are_refused():
    with pytest.raises(ValueError):
        pp_to_fraction(-1000)


def test_state_is_matched_on_whole_words():
    assert detect_state('How accurate was NS?') == 'neg_sembilan'
    assert detect_state('How accurate are the predictions?') is None     # "ns" inside a word
    assert detect_state('Which seats did Johor get wrong?') == 'johor'
    assert detect_state('Perak forecast') == 'perak'
    assert detect_state('what about Malacca') == 'melaka'


def test_scenario_detector_is_not_too_broad():
    assert is_scenario_question('What if turnout drops 10 points in N.01 Buloh Kasap?')
    assert is_scenario_question('If the majority shrinks in N.05, who wins?')
    assert not is_scenario_question('Tell me if the model beat the baseline')
    assert not is_scenario_question('What was the Johor accuracy?')


def test_empty_llm_reply_does_not_crash():
    assert message_text(SimpleNamespace(content=None), 'fallback') == 'fallback'
    assert message_text(SimpleNamespace(content='   '), 'fallback') == 'fallback'
    assert message_text(SimpleNamespace(content=' hi '), 'fallback') == 'hi'

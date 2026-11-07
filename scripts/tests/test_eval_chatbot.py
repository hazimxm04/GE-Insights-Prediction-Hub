"""Tests for scripts/eval_chatbot.py: the scoring and the progress handling, with no model calls."""
import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location('ev', Path(__file__).resolve().parents[1] / 'eval_chatbot.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def test_numeric_tokens_match_whole_numbers_only():
    assert ev.contains('6 seats flipped', '6')
    assert not ev.contains('16 seats flipped', '6')
    assert not ev.contains('0.6 seats', '6')
    assert ev.contains('Bukit Katil', 'bukit katil')


def test_content_scoring_rules():
    assert ev.score_content('48 of 56', '48|56', '', '') == (True, '')
    assert ev.score_content('48 only', '48|56', '', '') == (False, 'missing_content')
    assert ev.score_content('it is unavailable', '', 'no forecast|unavailable', '') == (True, '')
    assert ev.score_content('something else', '', 'no forecast|unavailable', '') == (False, 'missing_content')
    assert ev.score_content('high confidence in Melaka', '', '', 'high confidence') == (False, 'forbidden_phrase')


def test_tool_scoring():
    assert ev.score_tools('get_validation', ['get_validation', 'list_flips'])
    assert not ev.score_tools('get_validation|list_flips', ['get_validation'])
    assert ev.score_tools('', [])


def test_router_picks_one_tool_by_keywords():
    assert ev.route('Which seats flipped in Johor?') == 'list_flips'
    assert ev.route('How accurate was the model in Johor?') == 'get_validation'
    assert ev.route('What is the forecast for Melaka?') == 'get_state_forecast'
    assert ev.route('What if turnout fell?') == 'what_if_seat'
    assert ev.route('hello') is None


Q = {'id': 'x', 'category': 'validation', 'question': 'How accurate was Johor?', 'expected_tools': 'get_validation',
     'must_contain': '48|56', 'must_contain_any': '', 'must_not_contain': ''}
STEPS = [{'tool': 'get_validation', 'arguments': '{}', 'result': {'correct': 48, 'seats': 56}}]


def runner(answer, tools=('get_validation',), steps=STEPS, stopped='answered'):
    return lambda mode, question: {'answer': answer, 'tools_used': list(tools), 'steps': steps, 'stopped': stopped}


def test_a_good_answer_passes():
    r = ev.evaluate_one(Q, 'agent', runner('48 of 56'))
    assert r['pass'] and r['cause'] == ''


def test_failure_causes_are_named():
    assert ev.evaluate_one(Q, 'agent', runner('48 of 56', tools=()))['cause'] == 'wrong_tool'
    assert ev.evaluate_one(Q, 'agent', runner('about half'))['cause'] == 'missing_content'
    assert ev.evaluate_one(Q, 'agent', runner('48 of 56 and 99', ))['cause'] == 'unsupported_number'
    assert ev.evaluate_one(Q, 'agent', runner('x', stopped='error'))['cause'] == 'error'


def test_old_chatbot_is_not_judged_on_tools_or_numbers():
    r = ev.evaluate_one(Q, 'old', runner('48 of 56 and 99', tools=(), steps=[]))
    assert r['pass']


def test_rate_limit_detection():
    assert ev.is_rate_limit('Error code: 429 - rate_limit_exceeded')
    assert not ev.is_rate_limit('Seat not found')


def test_summary_counts(tmp_path):
    res = pd.DataFrame([
        {'id': 'a', 'mode': 'agent', 'category': 'v', 'pass': True, 'cause': ''},
        {'id': 'b', 'mode': 'agent', 'category': 'v', 'pass': False, 'cause': 'wrong_tool'},
        {'id': 'a', 'mode': 'router', 'category': 'v', 'pass': True, 'cause': ''}])
    text = ev.summarise(res)
    assert '| agent | 2 | 1 | 50% |' in text and 'wrong_tool' in text


def test_question_file_is_well_formed():
    q = pd.read_csv(ev.QUESTIONS, dtype=str).fillna('')
    assert {'id', 'question', 'expected_tools', 'must_contain', 'must_contain_any', 'must_not_contain',
            'requires_kb', 'category'} <= set(q.columns)
    assert q['id'].is_unique and len(q) >= 20
    from chatbot.agent import _registry
    known = {s['function']['name'] for s in _registry()[0]}
    for cell in q['expected_tools']:
        assert set(ev._split(cell)) <= known


def test_main_saves_progress_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, 'RESULTS', tmp_path / 'r.csv')
    monkeypatch.setattr(ev, 'SUMMARY', tmp_path / 's.md')
    calls = []

    def fake(mode, question, client=None, model=None):
        calls.append(question)
        return {'answer': 'x', 'tools_used': [], 'steps': [], 'stopped': 'answered'}

    monkeypatch.setattr(ev, 'run_mode', fake)
    monkeypatch.setattr(ev, 'evaluate_one', lambda q, m, runner=fake: ev.evaluate_one.__wrapped__(q, m, runner)
                        if hasattr(ev.evaluate_one, '__wrapped__') else {
                            'id': q['id'], 'mode': m, 'category': q['category'], 'answer': 'x', 'tools_used': '',
                            'content_pass': True, 'tool_pass': True, 'pass': True, 'cause': '', 'unsupported': '',
                            'seconds': 0})
    assert ev.main(['--modes', 'agent', '--limit', '3', '--sleep', '0']) == 0
    assert len(pd.read_csv(ev.RESULTS)) == 3 and (tmp_path / 's.md').exists()
    assert ev.main(['--modes', 'agent', '--limit', '5', '--sleep', '0']) == 0     # resumes: 2 new rows
    assert len(pd.read_csv(ev.RESULTS)) == 5


def test_summary_reports_plan_followed():
    res = pd.DataFrame([
        {'id': 'a', 'mode': 'agent_planned', 'category': 'v', 'pass': True, 'cause': '', 'plan_followed': '1.0'},
        {'id': 'b', 'mode': 'agent_planned', 'category': 'v', 'pass': True, 'cause': '', 'plan_followed': '0.5'}])
    assert '75% of the planned tool steps' in ev.summarise(res)

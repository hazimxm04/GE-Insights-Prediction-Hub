"""Tests for chatbot/verifier.py and the verify option of the agent loop."""
import json
from types import SimpleNamespace as NS

from chatbot import agent
from chatbot.verifier import check_answer

STEPS = [{'tool': 'get_validation', 'arguments': '{"state": "johor"}',
          'result': {'seats': 56, 'correct': 48, 'accuracy': 0.8571,
                     'baselines_correct': {'Opposition never wins': 48, 'Same result as previous election': 50},
                     'harapan_side_calls': {'made': 14, 'right': 7}}}]


def test_numbers_from_tools_pass():
    assert check_answer('The model got 48 of 56 seats right.', STEPS)['ok']


def test_percentage_of_a_tool_ratio_passes():
    assert check_answer('That is 85.7% accuracy.', STEPS)['ok']


def test_difference_of_two_tool_numbers_passes():
    assert check_answer('It was 2 seats behind the persistence rule.', STEPS)['ok']      # 50 - 48


def test_invented_number_fails():
    r = check_answer('The model got 52 of 56 seats right.', STEPS)
    assert not r['ok'] and r['unsupported'] == ['52']


def test_numbers_from_the_question_are_allowed():
    assert check_answer('Turnout 10 points lower.', [], question='what if turnout were 10 points lower')['ok']


def test_seat_ids_years_dates_and_list_markers_are_not_figures():
    ans = '1. N.12 Bentayan was right.\n2. In 2026 (frozen 2026-10-30) nothing changed.'
    assert check_answer(ans, [])['ok']


def test_answer_with_no_numbers_is_ok():
    r = check_answer('The forecast is not available yet.', [])
    assert r['ok'] and r['checked'] == 0


def test_numbers_with_no_tool_calls_fail():
    assert not check_answer('Harapan wins 32 seats.', [])['ok']


def test_numbers_inside_tool_strings_count():
    steps = [{'tool': 't', 'arguments': '{}', 'result': {'seats': [{'seat': 'N.17 Bukit Katil', 'p': 0.3609}]}}]
    assert check_answer('Bukit Katil has a 36.1% chance for Harapan.', steps)['ok']


# ── the loop ────────────────────────────────────────────────────────────────

class Fake:
    def __init__(self, replies):
        self.replies, self.sent = list(replies), []
        self.chat = NS(completions=NS(create=self._c))

    def _c(self, **kw):
        self.sent.append(kw)
        return self.replies.pop(0)


def _tool(name, args):
    return NS(choices=[NS(message=NS(content=None, tool_calls=[NS(id='c', function=NS(name=name, arguments=json.dumps(args)))]))])


def _text(t):
    return NS(choices=[NS(message=NS(content=t, tool_calls=None))])


def test_unverified_run_does_not_check():
    c = Fake([_tool('get_validation', {'state': 'johor'}), _text('Got 99 right')])
    r = agent.run_agent('johor q', client=c, model='m')
    assert 'verification' not in r and r['answer'] == 'Got 99 right'


def test_bad_number_is_sent_back_once_and_fixed():
    c = Fake([_tool('get_validation', {'state': 'johor'}), _text('It got 99 right.'), _text('It got 48 right.')])
    r = agent.run_agent('how did johor do', client=c, model='m', verify=True)
    assert r['answer'] == 'It got 48 right.' and r['verification']['retried'] and r['verification']['ok']
    assert '99' in c.sent[2]['messages'][-1]['content']          # the model was told which number


def test_still_wrong_after_retry_gets_a_warning():
    c = Fake([_tool('get_validation', {'state': 'johor'}), _text('77 right'), _text('76 right')])
    r = agent.run_agent('johor q', client=c, model='m', verify=True)
    assert 'Not verified' in r['answer'] and '76' in r['answer'] and not r['verification']['ok']


def test_good_answer_is_not_retried():
    c = Fake([_tool('get_validation', {'state': 'johor'}), _text('48 of 56')])
    r = agent.run_agent('johor q', client=c, model='m', verify=True)
    assert r['verification'] == {'ok': True, 'unsupported': [], 'checked': 2, 'retried': False} and len(c.sent) == 2


def test_a_wrong_year_is_caught():
    steps = [{'tool': 'list_flips', 'arguments': '{}', 'result': {'previous_election_year': 2022, 'count': 6}}]
    assert not check_answer('Compared with the 2021 election, 6 seats flipped.', steps)['ok']
    assert check_answer('Compared with the 2022 election, 6 seats flipped.', steps)['ok']
    assert check_answer('In the 2026 election, 6 seats flipped.', steps)['ok']
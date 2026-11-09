"""Tests for chatbot/agent.py with a fake model: no API key, no network."""
import json
from types import SimpleNamespace as NS

from chatbot import agent


def _call(name, args, cid='c1'):
    return NS(id=cid, function=NS(name=name, arguments=json.dumps(args)))


def _resp(content=None, calls=None):
    return NS(choices=[NS(message=NS(content=content, tool_calls=calls))])


class FakeClient:
    """Plays back scripted replies and records what it was sent."""
    def __init__(self, replies):
        self.replies, self.sent = list(replies), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.sent.append(kw)
        return self.replies.pop(0)


def test_answers_without_tools_when_none_needed():
    r = agent.run_agent('what is the model', client=FakeClient([_resp('Hello')]), model='m')
    assert r['answer'] == 'Hello' and r['steps'] == [] and r['stopped'] == 'answered'


def test_calls_a_tool_then_answers_with_its_result():
    c = FakeClient([_resp(calls=[_call('get_validation', {'state': 'johor'})]), _resp('Johor 48 of 56')])
    r = agent.run_agent('how did johor do', client=c, model='m')
    assert r['tools_used'] == ['get_validation'] and r['stopped'] == 'answered'
    assert r['steps'][0]['result'].get('available') is not False and 'error' not in r['steps'][0]['result']
    tool_msg = [m for m in c.sent[1]['messages'] if m['role'] == 'tool'][0]
    assert '48' in tool_msg['content']                      # the model was shown the real result


def test_can_chain_two_tools():
    c = FakeClient([_resp(calls=[_call('get_validation', {'state': 'johor'})]),
                    _resp(calls=[_call('list_flips', {'state': 'johor'})]), _resp('done')])
    r = agent.run_agent('johor q', client=c, model='m')
    assert r['tools_used'] == ['get_validation', 'list_flips']


def test_unavailable_forecast_is_passed_to_the_model_not_hidden():
    c = FakeClient([_resp(calls=[_call('get_state_forecast', {'state': 'johor'})]), _resp('none yet')])
    r = agent.run_agent('johor forecast?', client=c, model='m')
    assert r['steps'][0]['result']['available'] is False
    assert 'no forecast model yet' in r['steps'][0]['result']['reason']


def test_unknown_tool_and_bad_arguments_do_not_crash():
    c = FakeClient([_resp(calls=[_call('nope', {})]), _resp(calls=[_call('get_seat', {'wrong': 1})]), _resp('ok')])
    r = agent.run_agent('johor q', client=c, model='m')
    assert 'error' in r['steps'][0]['result'] and 'error' in r['steps'][1]['result']
    assert r['stopped'] == 'answered'


def test_stops_at_the_step_limit():
    c = FakeClient([_resp(calls=[_call('get_sentiment', {})]) for _ in range(10)])
    r = agent.run_agent('johor q', client=c, model='m', max_steps=3)
    assert r['stopped'] == 'step_limit' and len(r['steps']) == 3


def test_model_failure_is_reported_not_raised():
    class Boom:
        chat = NS(completions=NS(create=lambda **k: (_ for _ in ()).throw(RuntimeError('rate limit'))))
    r = agent.run_agent('johor q', client=Boom(), model='m')
    assert r['stopped'] == 'error' and 'rate limit' in r['answer']


def test_empty_model_answer_gets_a_message():
    r = agent.run_agent('johor q', client=FakeClient([_resp('')]), model='m')
    assert 'empty' in r['answer']


def test_history_is_sent_and_system_prompt_first():
    c = FakeClient([_resp('x')])
    agent.run_agent('johor q', client=c, model='m', history=[{'role': 'user', 'content': 'a'},
                                                       {'role': 'assistant', 'content': 'b'}])
    msgs = c.sent[0]['messages']
    assert msgs[0]['role'] == 'system' and [m['content'] for m in msgs[1:]] == ['a', 'b', 'johor q']


def test_long_tool_result_is_cut():
    assert len(agent._to_message_text({'x': 'a' * 20000})) < agent.MAX_TOOL_CHARS + 100


def test_extra_tool_can_be_added():
    schema = {'type': 'function', 'function': {'name': 'search_docs', 'description': 'd',
                                               'parameters': {'type': 'object', 'properties': {}}}}
    c = FakeClient([_resp(calls=[_call('search_docs', {})]), _resp('ok')])
    r = agent.run_agent('johor q', client=c, model='m', extra_tools=[(schema, lambda: {'hits': 1})])
    assert r['steps'][0]['result'] == {'hits': 1}
    assert any(t['function']['name'] == 'search_docs' for t in c.sent[0]['tools'])


def test_what_if_rejects_absurd_swing_and_bad_state():
    assert 'error' in agent.what_if_seat('johor', 'N.01', turnout_change_pp=90)
    assert 'error' in agent.what_if_seat('mars', 'N.01')


def test_api_key_is_read_from_the_environment_or_dotenv(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'abc')
    assert agent._api_key() == 'abc'

def test_tool_descriptions_and_prompt_keep_flips_apart_from_model_errors():
    from chatbot.agent import SYSTEM_PROMPT, _registry
    desc = {s['function']['name']: s['function']['description'] for s in _registry()[0]}
    assert 'NOT the list of seats the model got wrong' in desc['list_flips']
    assert 'wrong_seats' in desc['get_validation']
    assert 'get_validation (wrong_seats)' in SYSTEM_PROMPT and 'start with "No"' in SYSTEM_PROMPT
    assert 'the word "experimental"' in SYSTEM_PROMPT
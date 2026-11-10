"""Tests for chatbot/planner.py and plan=True in the agent loop (fake model, no network)."""
import json
from types import SimpleNamespace as NS

from chatbot import agent, planner as P

KNOWN = ['get_validation', 'list_flips', 'search_docs']


def plan_json(*steps, clarify=''):
    return json.dumps({'steps': [{'do': d, 'tool': t} for d, t in steps], 'clarify': clarify})


# ── reading a plan ──────────────────────────────────────────────────────────────

def test_good_plan_is_read():
    p = P.parse_plan(plan_json(('Get the Johor accuracy', 'get_validation'), ('List flips', 'list_flips')), KNOWN)
    assert p['ok'] and [s['tool'] for s in p['steps']] == ['get_validation', 'list_flips']


def test_json_wrapped_in_text_or_fences_is_still_read():
    wrapped = 'Sure!\n```json\n' + plan_json(('x', 'get_validation')) + '\n```'
    assert P.parse_plan(wrapped, KNOWN)['ok']


def test_unknown_tool_becomes_none_and_steps_are_capped():
    p = P.parse_plan(plan_json(*[(f'step {i}', 'made_up_tool') for i in range(9)]), KNOWN)
    assert len(p['steps']) == P.MAX_PLAN_STEPS and all(s['tool'] == 'none' for s in p['steps'])


def test_garbage_gives_an_empty_plan_not_an_error():
    for bad in ['', 'no json here', '{broken', '[]', None, '{"steps": "oops"}']:
        p = P.parse_plan(bad, KNOWN)
        assert p['steps'] == [] and not p['ok']


def test_clarify_only_counts_when_there_are_no_steps():
    assert P.parse_plan(plan_json(clarify='Which state?'), KNOWN)['clarify'] == 'Which state?'
    assert P.parse_plan(plan_json(('x', 'list_flips'), clarify='Which state?'), KNOWN)['clarify'] == ''


# ── tracking it ─────────────────────────────────────────────────────────────────

def test_progress_ticks_steps_in_order_and_handles_repeats():
    plan = P.parse_plan(plan_json(('a', 'get_validation'), ('b', 'get_validation'), ('c', 'list_flips')), KNOWN)
    got = P.mark_progress(plan, ['get_validation', 'list_flips'])
    assert [s['done'] for s in got] == [True, False, True]


def test_thinking_steps_tick_only_when_answered():
    plan = P.parse_plan(plan_json(('look up', 'get_validation'), ('compare', 'none')), KNOWN)
    assert [s['done'] for s in P.finish_plan(plan, ['get_validation'], answered=False)] == [True, False]
    assert [s['done'] for s in P.finish_plan(plan, ['get_validation'], answered=True)] == [True, True]


def test_followed_share():
    steps = [{'tool': 'a', 'done': True}, {'tool': 'b', 'done': False}, {'tool': 'none', 'done': False}]
    assert P.followed(steps) == 0.5 and P.followed([{'tool': 'none', 'done': False}]) == 1.0


def test_plan_text_lists_steps_and_is_empty_for_no_plan():
    assert P.plan_text({'steps': []}) == '' and P.plan_text(None) == ''
    t = P.plan_text(P.parse_plan(plan_json(('Get accuracy', 'get_validation')), KNOWN))
    assert '1. Get accuracy (tool: get_validation)' in t


def test_catalogue_names_every_tool():
    schemas, _ = agent._registry()
    cat = P.catalogue(schemas)
    assert all(s['function']['name'] in cat for s in schemas)


# ── in the loop ─────────────────────────────────────────────────────────────────

class Fake:
    def __init__(self, replies):
        self.replies, self.sent = list(replies), []
        self.chat = NS(completions=NS(create=self._c))

    def _c(self, **kw):
        self.sent.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def text(t):
    return NS(choices=[NS(message=NS(content=t, tool_calls=None))])


def tool(name, args=None):
    return NS(choices=[NS(message=NS(content=None, tool_calls=[
        NS(id='c', function=NS(name=name, arguments=json.dumps(args or {})))]))])


Q = 'How did the model do in Johor and which seats flipped?'


def test_plan_is_made_first_shown_followed_and_returned_ticked():
    plan = plan_json(('Get accuracy', 'get_validation'), ('List flips', 'list_flips'))
    c = Fake([text(plan), tool('get_validation', {'state': 'johor'}), tool('list_flips', {'state': 'johor'}),
              text('Done')])
    events = []
    r = agent.run_agent(Q, client=c, model='m', plan=True, on_event=lambda k, p: events.append(k))
    assert events == ['plan', 'tool', 'tool']
    assert [s['done'] for s in r['plan']] == [True, True] and r['plan_followed'] == 1.0
    assert c.sent[0].get('tools') is None                                  # the planner call has no tools
    assert any('Plan for this question' in m['content'] for m in c.sent[1]['messages'] if m['role'] == 'user')


def test_plan_followed_is_partial_when_the_agent_skips_a_step():
    plan = plan_json(('Get accuracy', 'get_validation'), ('List flips', 'list_flips'))
    c = Fake([text(plan), tool('list_flips', {'state': 'johor'}), text('Done')])
    r = agent.run_agent(Q, client=c, model='m', plan=True)
    assert [s['done'] for s in r['plan']] == [False, True] and r['plan_followed'] == 0.5


def test_unreadable_plan_means_the_agent_just_runs():
    c = Fake([text('lol no json'), text('Answer')])
    r = agent.run_agent(Q, client=c, model='m', plan=True)
    assert r['answer'] == 'Answer' and r['plan'] == []
    assert not any('Plan for this question' in m['content'] for m in c.sent[1]['messages'] if m['role'] == 'user')


def test_planner_failure_does_not_stop_the_answer():
    c = Fake([RuntimeError('rate limit'), text('Answer')])
    assert agent.run_agent(Q, client=c, model='m', plan=True)['answer'] == 'Answer'


def test_clarify_asks_without_running_the_loop():
    c = Fake([text(plan_json(clarify='Which state do you mean?'))])
    r = agent.run_agent('How accurate is the forecast model?', client=c, model='m', plan=True)
    assert r['stopped'] == 'clarify' and r['answer'] == 'Which state do you mean?' and len(c.sent) == 1


def test_no_plan_by_default_and_blocked_questions_are_not_planned():
    c = Fake([text('Answer')])
    assert 'plan' not in agent.run_agent(Q, client=c, model='m')
    c2 = Fake([])
    r = agent.run_agent('Give me a recipe for nasi lemak', client=c2, model='m', plan=True)
    assert r['stopped'] == 'blocked' and c2.sent == []


def test_a_broken_callback_never_stops_the_agent():
    def boom(kind, payload):
        raise RuntimeError('page closed')
    c = Fake([text(plan_json(('x', 'get_validation'))), tool('get_validation', {'state': 'johor'}), text('Done')])
    assert agent.run_agent(Q, client=c, model='m', plan=True, on_event=boom)['answer'] == 'Done'

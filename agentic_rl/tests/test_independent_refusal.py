"""Each clean current refusal is rewarded, independently of past/future labels."""
import json
from copy import deepcopy

import httpx
import pytest
from test_immediate_progress import config, row, turns
from test_simple_turn_reward import decision, inputs

from tau2_agentic_rl.advantages import compute_group_credit
from tau2_agentic_rl.reward.simple_turn import (
    build_label_trace,
    build_turn_inputs,
    turn_request,
    validate_decision,
)
from tau2_agentic_rl.schemas import ToolEvent

POLICY = {"violating_turns": [], "attribution_complete": True}


def independent_inputs(n=3):
    legacy = inputs(n)
    return build_turn_inputs(legacy['messages'], legacy['policy'], n, version='simple-turn-v2')


def judged(frozen, values):
    return [decision(t, request=turn_request(frozen, t, []), **kw) for t, kw in enumerate(values)]


def refusal():
    return dict(refusal=True, target='A', operation='cancel')


def test_every_repeated_clean_refusal_keeps_a_bonus():
    x = independent_inputs()
    ds = judged(x, [refusal(), refusal(), refusal()])
    trace = build_label_trace(x, ds, [], POLICY)
    assert [label['refusal_bonus'] for label in trace['turns']] == [1, 1, 1]
    r = row(phi=(0, 0, 0, 0))
    r['simple_turn_labels'] = trace
    result = compute_group_credit([r], config(final_turn_rule='simple-turn-v2'))
    assert turns(result['trajectories'][0]) == [1, 1, 1]


def test_future_violation_cannot_revoke_previous_refusal_or_influence_request():
    x = independent_inputs()
    ds = judged(x, [refusal(), refusal(), dict(bad=True)])
    before = deepcopy(ds)
    trace = build_label_trace(x, ds, [], POLICY)
    assert [label['eligible_correct_refusal'] for label in trace['turns']] == [True, True, False]
    req = turn_request(x, 1, ds[:1])
    assert req['previous_refusals'] == []
    assert len(req['history']) == x['turn_message_indices'][1]
    assert ds == before


def test_independent_mode_rejects_nonempty_retrospective_revocations():
    x = independent_inputs(2)
    ds = judged(x, [refusal(), dict(bad=True, revoked=[(0, 'A', 'cancel')])])
    with pytest.raises(ValueError, match='revocation'):
        build_label_trace(x, ds, [], POLICY)


def test_whole_turn_error_and_policy_ceiling_win_without_double_penalty():
    x = independent_inputs()
    ds = judged(x, [refusal(), refusal(), refusal()])
    events = [ToolEvent(event_id='ok', sequence=0, turn_id=1, success=True),
              ToolEvent(event_id='fail', sequence=1, turn_id=1, error_kind='parse_error')]
    trace = build_label_trace(x, ds, events, {**POLICY, 'violating_turns': [1]})
    r = row(phi=(0, 0, 0, 0), bad=(1,), costs=(.1, .08, 0))
    r['simple_turn_labels'] = trace
    value = compute_group_credit([r], config(final_turn_rule='simple-turn-v2'))['trajectories'][0]
    assert turns(value) == [-1, -1.08, 1]
    assert value['token_advantages'][2::3] == [0, 0, 0]


def test_multiple_refusals_in_one_generation_award_once_only():
    x = independent_inputs(1)
    ds = judged(x, [refusal()])
    ds[0]['refusal_requests'].append({'target': 'A', 'operation': 'change_flight'})
    trace = build_label_trace(x, ds, [], POLICY)
    assert trace['turns'][0]['refusal_bonus'] == 1
    r = row(phi=(0, 0))
    r['simple_turn_labels'] = trace
    assert turns(compute_group_credit([r], config(final_turn_rule='simple-turn-v2'))['trajectories'][0]) == [1]


def test_new_algorithm_rejects_legacy_eligibility_trace_instead_of_silently_reusing_it():
    ds = [decision(t) for t in range(3)]
    r = row()
    r['simple_turn_labels'] = build_label_trace(inputs(), ds, [], POLICY)
    with pytest.raises(ValueError, match='label'):
        compute_group_credit([r], config(final_turn_rule='simple-turn-v2'))


def test_truncation_zero_precedes_repeated_refusal_bad_and_group_baseline():
    x = independent_inputs()
    trace = build_label_trace(x, judged(x, [refusal(), refusal(), dict(bad=True)]), [], POLICY)
    r = row()
    r.update(score=0, multiplier=0, truncation_policy='zero-trajectory-v1',
             termination_reason='budget_exhausted', truncation_zero_applied=True,
             simple_turn_labels=trace)
    c = compute_group_credit([r], config(final_turn_rule='simple-turn-v2'))
    assert not c['has_signal']
    assert set(c['trajectories'][0]['token_advantages']) == {0}


def test_independent_label_still_rejects_future_evidence():
    x = independent_inputs()
    req = turn_request(x, 0, [])
    d = decision(0, request=req, **refusal())
    d['evidence_message_indices'].append(5)
    with pytest.raises(ValueError, match='evidence'):
        validate_decision(d, req)


@pytest.mark.asyncio
async def test_same_judge_topology_and_cache_separates_independent_from_legacy(monkeypatch, scratch_dir):
    from tau2_agentic_rl.judge.client import DeepSeekJudge, JudgeConfig
    from tau2_agentic_rl.judge.simple_turn import score_simple_turns
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'offline-fixture-key')
    calls = []
    def respond(wire):
        body = json.loads(wire.content)
        req = json.loads(body['messages'][1]['content'])
        calls.append(req)
        d = decision(req['turn_index'], request=req, **refusal())
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(d)}}]})
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: real(transport=httpx.MockTransport(respond), **kw))
    judge = DeepSeekJudge(JudgeConfig(model='fixture', cache_dir=str(scratch_dir)))
    old = await score_simple_turns(inputs(), [], POLICY, judge)
    x = independent_inputs()
    new = await score_simple_turns(x, [], POLICY, judge)
    assert len(calls) == 6  # Same three current-turn calls per mode; cache identities differ.
    assert [label['refusal_bonus'] for label in old['turns']] == [1, 0, 0]
    assert [label['refusal_bonus'] for label in new['turns']] == [1, 1, 1]
    assert all(not req['previous_refusals'] for req in calls[3:])
    assert await score_simple_turns(x, [], POLICY, judge) == new
    assert len(calls) == 6  # Identical new input reuses existing cache.

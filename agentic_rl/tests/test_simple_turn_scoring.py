import asyncio
import json
from copy import deepcopy

import httpx
import pytest
from test_procredit_rollout import rollout_fixture
from test_simple_turn_reward import decision, inputs, v4_row

from tau2_agentic_rl.advantages import CreditConfig
from tau2_agentic_rl.failures import JudgeServiceFailure
from tau2_agentic_rl.judge.client import DeepSeekJudge, JudgeConfig
from tau2_agentic_rl.procredit_runtime import build_credit_tensors, credit_from_record
from tau2_agentic_rl.reward.simple_turn import build_label_trace
from tau2_agentic_rl.scoring_retry import retry_scoring


async def fake_score(inputs, events, policy_credit, judge, **kwargs):
    turns = len(inputs['turn_message_indices'])
    decisions = [decision(t) for t in range(turns)]
    decisions[0] = decision(0, refusal=True, target='booking A', operation='remove_passenger')
    return build_label_trace(inputs, decisions, events, policy_credit)


def test_live_and_frozen_paths_keep_scalar_reward_and_original_trajectory(scratch_dir, monkeypatch):
    loop, output, record = rollout_fixture(scratch_dir, credit_version='procredit-turn-v4',
                                         final_turn_rule='simple-turn-v1', turn_scorer=fake_score)
    labels = record.metadata['simple_turn_labels']
    assert labels['turns'][0]['eligible_correct_refusal']
    assert labels['turns'][1]['bad_turn']
    assert output.extra_fields['procredit']['simple_turn_labels'] == labels
    before = deepcopy(record.model_dump())
    record.custom_reward = None
    record.metadata['failure_phase'] = 'turn_labels'
    monkeypatch.setattr('tau2_agentic_rl.scoring_retry.score_simple_turns', fake_score)
    assert asyncio.run(retry_scoring(record, loop.judge, loop.store))
    for key in ['messages', 'environment_transcript', 'token_turns', 'response_turn_ids',
                'scoring_inputs', 'custom_reward', 'trajectory_id']:
        assert record.model_dump()[key] == before[key]
    assert record.custom_reward.train_reward == output.reward_score
    record.metadata['simple_turn_labels']['turns'][0]['eligible_correct_refusal'] = False
    with pytest.raises(ValueError, match='labels'):
        credit_from_record(record)


def test_live_turn_judge_failure_retries_scoring_without_new_generation(scratch_dir, monkeypatch):
    async def fail(*args, **kwargs):
        raise JudgeServiceFailure('temporary turn judge failure', retryable=True)

    monkeypatch.setattr('tau2_agentic_rl.scoring_retry.score_simple_turns', fake_score)
    _, output, record = rollout_fixture(scratch_dir, credit_version='procredit-turn-v4',
                                      final_turn_rule='simple-turn-v1', turn_scorer=fail)
    assert record.metadata['scoring_retries'][0]['success']
    assert record.assistant_turns == 3
    assert output.extra_fields['procredit']['simple_turn_labels']['version'] == 'simple-turn-v1'


def test_cached_transport_retries_bad_evidence_and_never_sends_hidden_or_future_data(scratch_dir, monkeypatch):
    from tau2_agentic_rl.judge.simple_turn import score_simple_turns

    monkeypatch.setenv('TEST_JUDGE_KEY', 'fake-local-key')
    client = DeepSeekJudge(JudgeConfig(model='fixture', api_key_env='TEST_JUDGE_KEY',
                                     cache_dir=str(scratch_dir)))
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        data = json.loads(payload['messages'][1]['content'])
        calls.append(data)
        t = data['turn_index']
        d = decision(t)
        if len(calls) == 1:
            d['evidence_message_indices'] = [999]
        if t == 0:
            assert 'reply 1' not in str(payload)
        assert 'hidden tool result' not in str(payload)
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(d)}}]})

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: original(transport=httpx.MockTransport(respond), **kw))
    frozen = inputs(2)
    args = (frozen, [], {'violating_turns': [], 'attribution_complete': True}, client)
    first = asyncio.run(score_simple_turns(*args))
    assert len(calls) == 3
    assert first['turns'][0]['bad_turn'] is False
    assert len(first['judge_calls']) == 2
    assert asyncio.run(score_simple_turns(*args)) == first
    assert len(calls) == 3
    assert len(list((scratch_dir / 'simple_turn_v1').glob('*.json'))) == 2


def test_final_advantages_reach_actor_tokens_with_observation_and_padding_zero():
    torch = pytest.importorskip('torch')
    rows = [v4_row(i) for i in range(8)]
    trace = build_label_trace(inputs(2), [decision(0, bad=True),
                             decision(1, refusal=True, target='A', operation='cancel')], [],
                             {'violating_turns': [], 'attribution_complete': True})
    keys = [f'g_{i}_0' for i in range(8)] + ['padding_0_0']
    extras = [{'procredit': {**row, 'simple_turn_labels': trace}} for row in rows] + [{}]
    mask = torch.tensor([[1, 1, 0, 1, 1, 0]] * 8 + [[0] * 6])
    result, _ = build_credit_tensors(keys, extras, mask,
        CreditConfig(version='procredit-turn-v4', final_turn_rule='simple-turn-v1'),
        padding=[False] * 8 + [True])
    assert result['advantages'].tolist() == [[-1, -1, 0, 1, 1, 0]] * 8 + [[0] * 6]

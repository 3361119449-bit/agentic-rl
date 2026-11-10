import asyncio
from copy import deepcopy
from pathlib import Path

import pytest
from test_immediate_progress import config, row
from test_independent_refusal import POLICY, independent_inputs, judged, refusal
from test_procredit_rollout import rollout_fixture
from test_simple_turn_reward import decision
from test_simple_turn_scoring import fake_score

from scripts.train_airline_grpo import DEFAULT_CONFIG
from tau2_agentic_rl.advantages import CreditConfig, compute_group_credit
from tau2_agentic_rl.config import load_yaml, validate_procredit_config
from tau2_agentic_rl.procredit_runtime import build_credit_tensors, credit_from_record
from tau2_agentic_rl.reward.simple_turn import build_label_trace, turn_request
from tau2_agentic_rl.scoring_retry import retry_scoring

ROOT = Path(__file__).parents[1]


def test_live_freeze_retry_runtime_preserve_v2_and_original_tokens(scratch_dir, monkeypatch):
    async def scorer(inputs, events, policy_credit, judge, **kwargs):
        assert inputs['version'] == 'simple-turn-v2'
        return await fake_score(inputs, events, policy_credit, judge, **kwargs)
    loop, output, record = rollout_fixture(scratch_dir, credit_version='procredit-turn-v4',
                                           final_turn_rule='simple-turn-v2', turn_scorer=scorer)
    assert record.metadata['simple_turn_labels']['version'] == 'simple-turn-v2'
    assert output.extra_fields['procredit']['simple_turn_labels'] == record.metadata['simple_turn_labels']
    before = deepcopy(record.model_dump())
    record.custom_reward = None
    record.metadata['failure_phase'] = 'turn_labels'
    monkeypatch.setattr('tau2_agentic_rl.scoring_retry.score_simple_turns', scorer)
    assert asyncio.run(retry_scoring(record, loop.judge, loop.store))
    for key in ['messages', 'token_turns', 'response_turn_ids', 'environment_transcript',
                'scoring_inputs', 'trajectory_id', 'custom_reward']:
        assert record.model_dump()[key] == before[key]
    credit = credit_from_record(record)
    assert credit['simple_turn_labels']['version'] == 'simple-turn-v2'
    result = compute_group_credit([credit], config(final_turn_rule='simple-turn-v2'))
    assert result['has_signal']
    record.metadata['simple_turn_labels']['turns'][0]['eligible_correct_refusal'] = False
    with pytest.raises(ValueError, match='labels'):
        credit_from_record(record)


def test_delta_refusal_and_bad_signs_reach_real_actor_tensor_with_padding_zero():
    torch = pytest.importorskip('torch')
    x = independent_inputs()
    trace = build_label_trace(x, judged(x, [{}, refusal(), dict(bad=True)]), [], POLICY)
    rows = [{**row(i, phi=(0, 1, 1, 1)), 'simple_turn_labels': trace} for i in range(8)]
    keys = [f'g_{i}_0' for i in range(8)] + ['padding_0_0']
    extras = [{'procredit': r} for r in rows] + [{}]
    mask = torch.tensor([[1, 1, 0] * 3 + [0]] * 8 + [[0] * 10])
    result, _ = build_credit_tensors(keys, extras, mask, config(final_turn_rule='simple-turn-v2'),
                                      padding=[False] * 8 + [True])
    assert result['advantages'].tolist() == [[.5, .5, 0, 1, 1, 0, -1, -1, 0, 0]] * 8 + [[0] * 10]
    assert torch.equal(result['returns'], result['advantages'])


def test_current_configuration_drives_increment_and_independent_refusal():
    cfg = load_yaml(ROOT / DEFAULT_CONFIG)
    validate_procredit_config(cfg)
    x = independent_inputs()
    ds = [decision(t, request=turn_request(x, t, []), **({} if t == 0 else refusal())) for t in range(3)]
    r = row(phi=(0, .5, .5, .5))
    r['simple_turn_labels'] = build_label_trace(x, ds, [], POLICY)
    value = compute_group_credit([r], CreditConfig.from_project(cfg))['trajectories'][0]
    assert value['token_advantages'][::3] == [.25, 1, 1]


def test_current_configuration_changes_only_approved_selectors():
    cfg = load_yaml(ROOT / DEFAULT_CONFIG)
    old = load_yaml(ROOT / 'configs/rl/airline_procredit_v4_v12.yaml')
    assert cfg['project']['reward_version'] == 'v13-procredit-immediate-independent'
    assert cfg['credit'].pop('progress_credit') == 'immediate_delta_v1'
    assert cfg['credit']['turn_centering'] == 'none'
    assert cfg['credit']['final_turn_rule'] == 'simple-turn-v2'
    cfg['project']['reward_version'] = old['project']['reward_version']
    for key in ('turn_centering', 'final_turn_rule'):
        cfg['credit'][key] = old['credit'][key]
    assert cfg == old


def test_configuration_rejects_centering_and_extra_components_for_immediate_mode():
    cfg = load_yaml(ROOT / DEFAULT_CONFIG)
    cfg['credit'].update(progress_credit='immediate_delta_v1', turn_centering='policy_attributed_turn_mean')
    with pytest.raises(ValueError, match='turn_centering'):
        validate_procredit_config(cfg)
    cfg['credit']['turn_centering'] = 'none'
    for key, value in [('db_subgoal_credit', 'db_subgoal_local_v1'),
                       ('post_completion_credit', 'post_completion_regression_v1')]:
        bad = deepcopy(cfg)
        bad['credit'][key] = value
        with pytest.raises(ValueError, match='separate'):
            validate_procredit_config(bad)

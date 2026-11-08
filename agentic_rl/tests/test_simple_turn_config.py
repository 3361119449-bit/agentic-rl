from pathlib import Path

import pytest

from scripts.train_airline_grpo import DEFAULT_CONFIG
from tau2_agentic_rl.advantages import CreditConfig
from tau2_agentic_rl.config import load_yaml, validate_procredit_config

ROOT = Path(__file__).parents[1]


def test_default_enables_final_rule_and_legacy_keeps_original_config():
    cfg = load_yaml(ROOT / DEFAULT_CONFIG)
    legacy = load_yaml(ROOT / 'configs/rl/airline_procredit_v4_legacy.yaml')
    assert CreditConfig.from_project(cfg).final_turn_rule == 'simple-turn-v1'
    assert CreditConfig.from_project(legacy).final_turn_rule is None
    validate_procredit_config(cfg)
    assert cfg['judge'].pop('grounding_version') == 'judge-grounded-v3'
    # The v11 default adds only the three approved, versioned reward changes.
    assert cfg['reward'].pop('transfer_validation_version') == 'transfer-policy-v2'
    assert cfg['reward'].pop('duplicate_query_version') == 'duplicate-query-v1'
    assert cfg['reward']['progress_version'] == 'progress-v3'
    cfg['reward']['progress_version'] = 'progress-v2'
    cfg['credit'].pop('final_turn_rule')
    cfg['project']['reward_version'] = legacy['project']['reward_version']
    # New runs use eight-way rollout; the frozen legacy config stays at two.
    for key in ('agent_worker_count', 'max_active_trajectories',
                'user_api_max_inflight', 'judge_api_max_inflight', 'vllm_max_num_seqs'):
        assert cfg['rollout'].pop(key) == 8
        assert legacy['rollout'].pop(key) == 2
    # Current training/evaluation use30 assistant turns; legacy stays frozen.
    assert cfg['rollout']['max_hard_turns'] == 35
    assert cfg['rollout']['max_soft_turns'] == 35
    cfg['rollout']['max_soft_turns'] = legacy['rollout']['max_soft_turns']
    assert cfg['reward'].pop('truncation_policy') == 'zero-trajectory-v1'
    assert cfg['reward']['truncation_multiplier'] == 0
    cfg['reward']['truncation_multiplier'] = legacy['reward']['truncation_multiplier']
    assert cfg['slot_recovery'].pop('max_truncation_resamples') == 2
    assert legacy['rollout']['max_hard_turns'] == 24
    cfg['rollout']['max_hard_turns'] = legacy['rollout']['max_hard_turns']
    assert cfg == legacy


@pytest.mark.parametrize('mode,version', [('official', 'procredit-turn-v4'),
                                        ('strict_progress_v1', 'procredit-turn-v2')])
def test_rule_cannot_be_enabled_without_v4(mode, version):
    cfg = load_yaml(ROOT / 'configs/rl/airline_procredit_v4.yaml')
    cfg['reward']['mode'] = mode
    cfg['credit'].update(version=version, final_turn_rule='simple-turn-v1')
    with pytest.raises(ValueError):
        validate_procredit_config(cfg)

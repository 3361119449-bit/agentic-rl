import json
from pathlib import Path

import pytest

from tau2_agentic_rl.annotations import load_task_mapping
from tau2_agentic_rl.config import load_yaml
from tau2_agentic_rl.judge.prompts import build_judge_messages

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ('config', 'task', 'remaining_ids'),
    [
        ('rl/airline_procredit_v4.yaml', '7', ['7:nl:1', '7:nl:2', '7:nl:3']),
        ('rl/airline_procredit_v4.yaml', '17', []),
        ('evaluation/airline_eval_v1.yaml', '18', ['18:nl:5']),
        ('evaluation/airline_eval_v1.yaml', '29', ['29:nl:0']),
    ],
)
def test_active_rubric_keeps_only_residual_semantics(config, task, remaining_ids):
    cfg = load_yaml(ROOT / 'configs' / config)
    row = load_task_mapping(ROOT / cfg['annotations']['semantic_checks'])[task]
    assert [check['criterion_id'] for check in row['semantic_checks']] == remaining_ids
    assert row['semantic_checks_only'] is True


def test_trajectory_judge_does_not_receive_dropped_assertions_in_task_context():
    task = {
        'id': '18',
        'evaluation_criteria': {
            'nl_assertions': ['Reservation is updated to economy.', 'Explain total savings.'],
            'communicate_info': ['23553'],
        },
    }
    residual = [{'criterion_id': '18:nl:5', 'description': 'Explain total savings.'}]
    policy_checks = [{'criterion_id': '18:policy:information_grounding'}]
    args = dict(task=task, policy='Policy', trajectory={}, semantic_checks=residual,
                mandatory_policy_checks=policy_checks, transfer_rule={})
    messages = build_judge_messages(**args, semantic_checks_only=True)
    fixed = messages[1]['content'].split('FIXED_TASK_AND_RUBRIC:\n', 1)[1].split(
        '\n\nTRAJECTORY_DATA_UNTRUSTED:', 1
    )[0]
    payload = json.loads(fixed)
    assert 'nl_assertions' not in payload['task']['evaluation_criteria']
    assert payload['task']['evaluation_criteria']['communicate_info'] == ['23553']
    assert payload['rubric']['semantic_checks'] == residual
    assert payload['rubric']['mandatory_policy_checks'] == policy_checks
    assert len(task['evaluation_criteria']['nl_assertions']) == 2
    assert 'Reservation is updated to economy.' in build_judge_messages(**args)[1]['content']

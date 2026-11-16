import asyncio
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_immediate_progress import config, row
from test_independent_refusal import POLICY, independent_inputs, judged, refusal
from test_procredit_rollout import rollout_fixture
from test_simple_turn_reward import decision
from test_simple_turn_scoring import fake_score

from scripts.train_airline_grpo import DEFAULT_CONFIG, FULL_TRAIN_CONFIG
from tau2_agentic_rl.advantages import CreditConfig, compute_group_credit
from tau2_agentic_rl.config import load_yaml, validate_procredit_config
from tau2_agentic_rl.judge.prompts import rubric_fingerprint
from tau2_agentic_rl.procredit_runtime import build_credit_tensors, credit_from_record
from tau2_agentic_rl.reward.progress import build_progress_trace
from tau2_agentic_rl.reward.score import build_reward_config, score_trajectory
from tau2_agentic_rl.reward.simple_turn import build_label_trace, turn_request
from tau2_agentic_rl.schemas import (
    JudgeCheck,
    JudgeResult,
    OfficialScores,
    ToolEvent,
    TransferCheck,
)
from tau2_agentic_rl.scoring_retry import retry_scoring

ROOT = Path(__file__).parents[1]
OFFICIAL_PROGRESS = "official_semantic_progress_v1"


def test_train30_uses_official_completion_and_all_terminal_progress_checks():
    cfg = load_yaml(ROOT / FULL_TRAIN_CONFIG)
    validate_procredit_config(cfg)
    assert cfg["reward"]["task_score_mode"] == OFFICIAL_PROGRESS
    required = [{"action_id": "cancel", "name": "cancel_reservation",
                 "arguments": {"reservation_id": "R"}}]
    event = ToolEvent(event_id="cancel", sequence=0, turn_id=1, name="cancel_reservation",
                      arguments={"reservation_id": "R"}, success=True, db_effect=True)
    trace = build_progress_trace(
        task={"id": "test"}, messages=[{}], prefix_lengths=[0, 1],
        events=[event], required_actions=required, dependencies=[], transfer_rule={},
        initial_state_fingerprint="initial", version="progress-v3",
        evaluator=lambda task, messages: {"db": True, "communicate": []},
    )
    judge = JudgeResult(
        semantic_checks=[
            JudgeCheck(criterion_id="test:nl:0", passed=True),
            JudgeCheck(criterion_id="test:nl:1", passed=False),
        ],
        transfer_check=TransferCheck(applicable=False, valid=False, policy_required=False,
                                     self_caused_only=False, short_reason="No transfer needed."),
    )
    result = score_trajectory(
        events=[event], messages=[{}], assistant_turns=1, required_actions=required,
        official=OfficialScores(reward=1, db_applicable=True, db_score=1),
        judge=judge, config=build_reward_config(cfg), progress_trace=trace,
    )
    # Initial DB is excluded. One required action and one of two semantic checks passed.
    assert trace["phi"] == [0, 1]
    assert result.details["task_completion"] == result.strict_success == 1
    assert result.progress == pytest.approx(2 / 3)
    assert result.train_reward == pytest.approx(1 + 0.5 * (2 / 3))


def test_official_progress_retries_frozen_and_reaches_group_credit(scratch_dir):
    definitions = [{"criterion_id": "required-explanation", "criterion": "Explain."}]

    async def judge(**kwargs):
        assert kwargs["semantic_checks"] == definitions
        result = JudgeResult(semantic_checks=[
            JudgeCheck(criterion_id="required-explanation", passed=False),
        ])
        return result, "", "", ""

    loop, output, record = rollout_fixture(
        scratch_dir, credit_version="procredit-turn-v4", natural_stop_at=1,
        actual_judge=SimpleNamespace(evaluate=judge),
        semantic_row={"semantic_checks": definitions}, task_score_mode=OFFICIAL_PROGRESS,
    )
    row = credit_from_record(record)
    assert row == output.extra_fields["procredit"]
    assert row["terminal_success"] == record.official_scores.reward == 1
    assert row["phi"] == [0, 1]
    assert row["final_progress"] == .5
    assert row["score"] == 1.25
    worse = dict(row, trajectory_id="worse", final_progress=0, score=1)
    report = compute_group_credit([row, worse], config())
    assert report["score_mean"] == 1.125
    assert report["trajectories"][0]["turn_returns"] == [.5]
    assert report["trajectories"][0]["trajectory_advantage"] > 0
    assert report["trajectories"][1]["trajectory_advantage"] < 0

    record.custom_reward = None
    record.metadata["failure_phase"] = "judge"
    assert asyncio.run(retry_scoring(record, loop.judge, loop.store))
    assert credit_from_record(record) == row
    record.judge_result.semantic_checks[0].passed = True
    with pytest.raises(ValueError, match="reward differs"):
        credit_from_record(record)


def test_new_credit_keeps_partial_official_scores_and_rejects_mixing_versions():
    value = row(phi=(0, .5), terminal=.25)
    value.update(final_progress=.75, score=.625)
    assert compute_group_credit([value], config())["score_mean"] == .625
    with pytest.raises(ValueError, match="mixed group identity"):
        compute_group_credit([value, row(1, phi=(0, .5))], config())
    with pytest.raises(ValueError, match="scalar score differs"):
        compute_group_credit([dict(value, score=.5)], config())


def test_live_freeze_retry_runtime_preserve_v2_and_original_tokens(scratch_dir, monkeypatch):
    guidance = "Check each reservation independently."

    async def scorer(inputs, events, policy_credit, judge, **kwargs):
        assert inputs['version'] == 'simple-turn-v2'
        assert inputs['task_guidance'] == guidance
        return await fake_score(inputs, events, policy_credit, judge, **kwargs)
    loop, output, record = rollout_fixture(scratch_dir, credit_version='procredit-turn-v4',
                                           final_turn_rule='simple-turn-v2', turn_scorer=scorer,
                                           semantic_row={'semantic_checks_only': True},
                                           task_guidance=guidance)
    assert record.scoring_inputs['judge']['semantic_checks_only'] is True
    assert record.scoring_inputs['judge']['task_guidance'] == guidance
    assert record.metadata['judge_rubric_sha256'] == rubric_fingerprint(
        [], [], {}, require_policy_attribution=True, semantic_checks_only=True,
        task_guidance=guidance,
    )
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
    assert cfg['project']['reward_version'] == 'v13.1-procredit-immediate-independent'
    assert cfg['reward'].pop('check_reference_write_scope') is False
    assert cfg['credit'].pop('progress_credit') == 'immediate_delta_v1'
    assert cfg['credit']['turn_centering'] == 'none'
    assert cfg['credit']['final_turn_rule'] == 'simple-turn-v2'
    assert cfg['project']['annotation_version'] == 'v6-task-judge-guidance'
    assert cfg['annotations']['semantic_checks'].endswith('.train.v2.json')
    assert cfg['annotations'].pop('task_judge_guidance').endswith('.train.v1.json')
    cfg['project']['reward_version'] = old['project']['reward_version']
    cfg['project']['annotation_version'] = old['project']['annotation_version']
    cfg['annotations']['semantic_checks'] = old['annotations']['semantic_checks']
    for key in ('turn_centering', 'final_turn_rule'):
        cfg['credit'][key] = old['credit'][key]
    assert cfg == old


def test_reference_write_scope_does_not_punish_authorized_baggage():
    cfg = load_yaml(ROOT / DEFAULT_CONFIG)
    required = [{"action_id": "rebook", "name": "book_reservation",
                 "arguments": {"user_id": "traveler", "total_baggages": 1}}]
    events = [
        ToolEvent(event_id="change", sequence=0, turn_id=1,
                  name="update_reservation_flights", arguments={"reservation_id": "R"},
                  success=True, db_effect=True),
        ToolEvent(event_id="baggage", sequence=1, turn_id=2,
                  name="update_reservation_baggages",
                  arguments={"reservation_id": "R", "total_baggages": 1,
                             "nonfree_baggages": 0}, success=True, db_effect=True),
        ToolEvent(event_id="partial", sequence=2, turn_id=3,
                  name="update_reservation_baggages",
                  arguments={"reservation_id": "R"}, success=False, db_effect=True),
    ]
    trace = build_progress_trace(
        task={"id": "29"}, messages=[{}, {}, {}], prefix_lengths=[0, 1, 2, 3],
        events=events, required_actions=required, dependencies=[], transfer_rule={},
        initial_state_fingerprint="initial", version="progress-v3",
        evaluator=lambda task, messages: {"db": False, "communicate": []},
    )
    judge = JudgeResult(
        mandatory_policy_checks=[JudgeCheck(
            criterion_id="29:policy:flight_change_eligibility", passed=False,
            violation_assistant_turn_ids=[1], short_reason="Destination change on existing booking.",
        )],
        transfer_check=TransferCheck(applicable=False, valid=False, policy_required=False,
                                     self_caused_only=False, short_reason="No transfer needed."),
    )
    inputs = dict(events=events, messages=[], assistant_turns=3, required_actions=required,
                  official=OfficialScores(reward=0, db_applicable=True, db_score=0),
                  judge=judge, progress_trace=trace)
    reward = score_trajectory(config=build_reward_config(cfg), **inputs)
    historical = deepcopy(cfg)
    del historical["reward"]["check_reference_write_scope"]
    before = score_trajectory(config=build_reward_config(historical), **inputs)
    assert before.details["policy_credit"]["violating_turns"] == [0, 1, 2]
    assert reward.details["policy_credit"]["violating_turns"] == [0, 2]
    assert reward.details["policy_turn_rewards"] == [-1.0, 0.0, -1.0]
    assert reward.details["task_completion"] == before.details["task_completion"] == 0
    assert reward.train_reward == before.train_reward == 0


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

import pytest

from tau2_agentic_rl.reward.score import build_reward_config, score_trajectory
from tau2_agentic_rl.schemas import JudgeCheck, JudgeResult, OfficialScores


def test_partial_components_are_weighted_and_inapplicable_items_omitted() -> None:
    result = score_trajectory(
        events=[],
        messages=[],
        assistant_turns=1,
        required_actions=[],
        official=OfficialScores(
            reward=0.0,
            db_applicable=True,
            db_score=1.0,
            communicate_applicable=True,
            communicate_partial=0.5,
        ),
        judge=JudgeResult(semantic_checks=[JudgeCheck(criterion_id="s", passed=False)]),
    )
    # (0.30*1 + 0.15*0.5 + 0.25*0) / 0.70, then 0.75 progress.
    assert result.progress == pytest.approx(0.375 / 0.70)
    assert result.train_reward == pytest.approx(0.75 * 0.375 / 0.70)
    assert result.strict_success == 0.0


def test_all_applicable_components_pass_gives_one() -> None:
    result = score_trajectory(
        events=[],
        messages=[],
        assistant_turns=1,
        required_actions=[],
        official=OfficialScores(reward=1, db_applicable=True, db_score=1),
        judge=JudgeResult(),
    )
    assert result.train_reward == 1.0
    assert result.strict_success == 1.0


def test_yaml_weight_mapping_changes_frozen_trajectory_score() -> None:
    inputs = dict(
        events=[],
        messages=[],
        assistant_turns=1,
        required_actions=[],
        official=OfficialScores(reward=0, db_applicable=True, db_score=1),
        judge=JudgeResult(semantic_checks=[JudgeCheck(criterion_id="s", passed=False)]),
    )
    baseline = score_trajectory(**inputs)
    configured = score_trajectory(
        **inputs,
        config=build_reward_config(
            {
                "rollout": {"max_soft_turns": 9},
                "reward": {
                    "normal_weights": {"db": 0, "judge_semantic": 1},
                    "progress_coefficient": 1,
                    "strict_success_coefficient": 0,
                    "process_penalty_cap": 0.2,
                },
            }
        ),
    )
    assert baseline.train_reward > 0
    assert configured.train_reward == 0


def test_yaml_gate_switches_are_wired_into_reward_config() -> None:
    config = build_reward_config(
        {"reward": {"mandatory_policy_gate": False, "task_safety_gate": False}}
    )
    assert config.enable_mandatory_policy_gate is False
    assert config.enable_task_safety_gate is False

import pytest
from pydantic import ValidationError

from tau2_agentic_rl.reward.progress import build_progress_trace
from tau2_agentic_rl.reward.score import build_reward_config, score_trajectory
from tau2_agentic_rl.schemas import JudgeResult, OfficialScores, RewardResult


def trace(phi=1.0, turns=16):
    return build_progress_trace(
        task={"id": "0"},
        messages=[{}] * (turns + 1),
        prefix_lengths=list(range(1, turns + 2)),
        events=[],
        required_actions=[],
        dependencies=[],
        transfer_rule={},
        initial_state_fingerprint="initial",
        evaluator=lambda task, messages: {
            "db": len(messages) > 1 and bool(phi),
            "communicate": [],
        },
    )


def inputs():
    return dict(
        events=[],
        messages=[],
        assistant_turns=16,
        required_actions=[],
        official=OfficialScores(reward=1, db_applicable=True, db_score=1),
        judge=JudgeResult(),
        config=build_reward_config({"reward": {"mode": "strict_progress_v1"}}),
        progress_trace=trace(),
    )


def test_new_score_survives_schema_and_penalty_outside_truncation():
    result = score_trajectory(**inputs(), termination_reason="generation_truncated")
    assert result.train_reward == pytest.approx(1.105)
    assert result.strict_success == 1
    assert result.process_penalty == pytest.approx(0.02)
    assert result.reward_mode == "strict_progress_v1"
    restored = RewardResult.model_validate_json(result.model_dump_json())
    assert restored.train_reward == result.train_reward
    assert result.details["score_before_floor"] == pytest.approx(1.105)


def test_normal_success_reaches_one_point_five_without_clipping():
    kwargs = inputs()
    kwargs["assistant_turns"] = 2
    kwargs["progress_trace"] = trace(turns=2)
    result = score_trajectory(**kwargs)
    assert result.train_reward == 1.5
    with pytest.raises(ValidationError):
        RewardResult.model_validate({**result.model_dump(), "reward_mode": "legacy"})


def test_new_mode_requires_complete_trace():
    kwargs = inputs()
    kwargs.pop("progress_trace")
    with pytest.raises(ValueError, match="progress"):
        score_trajectory(**kwargs)


def test_new_reward_rejects_progress_not_supported_by_checks():
    kwargs = inputs()
    kwargs["progress_trace"]["check_values"][-1] = [False]
    with pytest.raises(ValueError, match="progress"):
        score_trajectory(**kwargs)


def test_changed_required_actions_cannot_reuse_old_progress():
    kwargs = inputs()
    kwargs["required_actions"] = [
        {"action_id": "new", "name": "get_user_details", "arguments": {"user_id": "a"}}
    ]
    with pytest.raises(ValueError, match="progress"):
        score_trajectory(**kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "normal_weights",
        "transfer_weights",
        "progress_coefficient",
        "strict_success_coefficient",
    ],
)
def test_new_mode_rejects_silently_ignored_legacy_weights(field):
    with pytest.raises(ValueError, match="legacy"):
        build_reward_config({"reward": {"mode": "strict_progress_v1", field: 0.5}})

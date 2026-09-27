from copy import deepcopy

import pytest

from tau2_agentic_rl import advantages


def row(index=0, **changes):
    result = {
        "trajectory_id": f"trajectory-{index}",
        "task_id": "task",
        "policy_version": 1,
        "checkset_fingerprint": "checks",
        "initial_state_fingerprint": "initial",
        "score": 1.5,
        "valid": True,
        "terminal_success": 1.0,
        "multiplier": 1.0,
        "phi": [0.0, 1.0, 1.0],
        "response_turn_ids": [0, 0, -1, 1, 1],
    }
    result.update(changes)
    return result


def test_equal_scores_keep_turn_credit_and_mask_observations():
    result = advantages.compute_group_credit([row(i) for i in range(8)])
    assert result["has_signal"]
    assert result["score_std"] == 0
    assert result["turn_mean"] == 1.25
    for item in result["trajectories"]:
        assert item["trajectory_advantage"] == 0
        assert item["turn_advantages"] == [0.25, -0.25]
        assert item["token_advantages"] == [0.25, 0.25, 0, -0.25, -0.25]


def test_center_counts_turns_not_tokens_or_trajectories():
    rows = [
        row(0, score=0.5, terminal_success=0, phi=[0, 1], response_turn_ids=[0] * 20),
        row(
            1,
            score=0.5,
            terminal_success=0,
            phi=[0, 0, 1, 1],
            response_turn_ids=[0, 1, 2],
        ),
    ]
    result = advantages.compute_group_credit(rows)
    assert result["turn_mean"] == 0.375
    assert result["trajectories"][0]["turn_advantages"] == [0.125]
    assert result["trajectories"][1]["turn_advantages"] == [0.125, 0.125, -0.375]


def test_invalid_trajectory_never_has_positive_credit_even_with_regression():
    result = advantages.compute_group_credit(
        [
            row(0, score=0, valid=False, terminal_success=0, phi=[0, 1, 0]),
            row(1, score=0.2, terminal_success=0, phi=[0, 1, 0.4]),
        ]
    )
    invalid = result["trajectories"][0]
    assert invalid["turn_advantages"] == [0, 0]
    assert max(invalid["token_advantages"]) == 0  # observation token
    assert invalid["trajectory_advantage"] < 0
    assert result["turn_mean"] == pytest.approx(-0.05)


def test_all_invalid_has_no_signal_and_no_nan():
    result = advantages.compute_group_credit(
        [row(i, score=0, valid=False, terminal_success=0) for i in range(8)]
    )
    assert result["turn_mean"] == 0
    assert not result["has_signal"]
    assert result["trajectories"][0]["token_advantages"] == [0] * 5


def test_task_regression_matches_adjacent_credit_difference():
    result = advantages.compute_group_credit(
        [
            row(i, score=0, terminal_success=0, multiplier=0.75, phi=[0, 1, 0])
            for i in range(8)
        ]
    )
    assert result["trajectories"][0]["turn_advantages"] == [0.1875, -0.1875]


def test_zero_turn_coefficient_is_scalar_ablation():
    cfg = advantages.CreditConfig(turn_coefficient=0)
    result = advantages.compute_group_credit([row(i) for i in range(8)], cfg)
    assert not result["has_signal"]
    assert result["trajectories"][0]["token_advantages"] == [0] * 5


@pytest.mark.parametrize(
    "change",
    [
        {"score": float("nan")},
        {"score": 1.6},
        {"phi": [0, 2, 1]},
        {"phi": [0.2, 1, 1]},
        {"response_turn_ids": [0, 2]},
        {"response_turn_ids": [True, 1]},
        {"response_turn_ids": [-1, -1]},
        {"valid": False, "score": 0.2},
        {"terminal_success": 0.5},
    ],
)
def test_malformed_row_is_error_not_zero_reward(change):
    with pytest.raises(ValueError):
        advantages.compute_group_credit([row(0, **change), row(1)])


@pytest.mark.parametrize(
    "field",
    [
        "task_id",
        "policy_version",
        "checkset_fingerprint",
        "initial_state_fingerprint",
    ],
)
def test_mixed_group_identity_rejected(field):
    rows = [row(0), row(1)]
    rows[1][field] = 2 if field == "policy_version" else "different"
    with pytest.raises(ValueError, match="identity"):
        advantages.compute_group_credit(rows)


def test_duplicate_member_rejected_and_inputs_not_mutated():
    original = row()
    before = deepcopy(original)
    with pytest.raises(ValueError, match="duplicate"):
        advantages.compute_group_credit([original, original])
    assert original == before

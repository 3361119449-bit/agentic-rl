from copy import deepcopy

import pytest
from test_procredit_credit import row

from tau2_agentic_rl.advantages import CreditConfig, compute_group_credit


def policy_row(i=0, **changes):
    result = row(
        i,
        valid=False,
        score=0,
        terminal_success=0,
        phi=[0, 0.5, 1, 1],
        response_turn_ids=[0, 0, -1, 1, 2],
    )
    result["policy_credit"] = {
        "version": "policy-credit-v1",
        "violating_turns": [2],
        "attribution_complete": True,
        "unresolved_checks": [],
    }
    result.update(changes)
    return result


def v2(**kwargs):
    return CreditConfig(version="procredit-turn-v2", **kwargs)


def test_all_zero_terminal_scores_keep_verified_progress_and_penalize_bad_turn():
    result = compute_group_credit([policy_row(i) for i in range(8)], v2())
    assert result["has_signal"]
    assert result["score_std"] == 0
    assert result["trajectories"][0]["turn_advantages"] == [0.25, 0, -1.25]
    assert result["trajectories"][0]["token_advantages"] == [0.25, 0.25, 0, 0, -1.25]


def test_zero_progress_group_has_negative_feedback_without_inventing_positive_progress():
    rows = [policy_row(i, phi=[0, 0, 0, 0]) for i in range(8)]
    result = compute_group_credit(rows, v2())
    assert result["has_signal"]
    assert result["trajectories"][0]["turn_advantages"] == [0, 0, -1]


def test_progress_created_on_policy_violating_turn_is_not_rewarded():
    rows = [policy_row(i, phi=[0, 0, 0, 1]) for i in range(8)]
    result = compute_group_credit(rows, v2())
    assert result["trajectories"][0]["turn_advantages"] == [0, 0, -1]


def test_unresolved_blame_does_not_spread_punishment_or_reward_earlier_steps():
    rows = [policy_row(i) for i in range(8)]
    for item in rows:
        item["policy_credit"].update(
            attribution_complete=False, unresolved_checks=["information_grounding"]
        )
    result = compute_group_credit(rows, v2())
    assert result["trajectories"][0]["turn_advantages"] == [0, 0, -1]
    assert result["unresolved_trajectories"] == 8


def test_v1_original_group_math_and_serialized_config_remain_unchanged():
    rows = [policy_row(i) for i in range(8)]
    for item in rows:
        item.pop("policy_credit")
    result = compute_group_credit(rows)
    assert not result["has_signal"]
    assert set(result["config"]) == {
        "progress_scale",
        "turn_coefficient",
        "epsilon",
        "signal_tolerance",
    }
    assert not compute_group_credit(
        [policy_row(i) for i in range(8)], v2(turn_coefficient=0)
    )["has_signal"]


@pytest.mark.parametrize(
    "policy",
    [
        None,
        {},
        {
            "version": "policy-credit-v1",
            "violating_turns": [99],
            "attribution_complete": True,
            "unresolved_checks": [],
        },
    ],
)
def test_v2_cannot_silently_train_without_complete_attribution_metadata(policy):
    rows = [policy_row(i, policy_credit=deepcopy(policy)) for i in range(8)]
    with pytest.raises(ValueError):
        compute_group_credit(rows, v2())

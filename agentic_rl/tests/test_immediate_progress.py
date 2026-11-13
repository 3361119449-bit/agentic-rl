"""Independent hand-calculated expectations for immediate progress credit."""
from copy import deepcopy

import pytest

from tau2_agentic_rl.advantages import CreditConfig, compute_group_credit


def row(index=0, *, phi=(0, .5, 1, 1), terminal=1, bad=(), costs=None):
    n = len(phi) - 1
    return dict(
        trajectory_id=f"delta-{index}", task_id="task", policy_version=0,
        checkset_fingerprint="checks", initial_state_fingerprint="initial",
        score=terminal + .5 * phi[-1], terminal_success=terminal, valid=not bad,
        multiplier=1, phi=list(phi), credit_version="procredit-turn-v4",
        response_turn_ids=[x for t in range(n) for x in (t, t, -1)],
        policy_credit={"version": "policy-credit-v1", "violating_turns": list(bad),
                       "attribution_complete": True, "unresolved_checks": []},
        process_credit={"version": "process-credit-v2",
                        "turn_costs": list(costs) if costs is not None else [0.] * n},
    )


def config(**kwargs):
    return CreditConfig(version="procredit-turn-v4", progress_credit="immediate_delta_v1", **kwargs)


def turns(result):
    return result["token_advantages"][::3]


def test_equal_success_scores_reward_only_current_progress_and_keep_masks():
    rows = [row(i) for i in range(4)]
    before = deepcopy(rows)
    report = compute_group_credit(rows, config())
    for value in report["trajectories"]:
        assert value["trajectory_advantage"] == 0
        assert value["turn_returns"] == [.25, .25, 0]
        assert turns(value) == [.25, .25, 0]
        assert value["token_advantages"][2::3] == [0, 0, 0]
    assert report["has_signal"]
    assert report["turn_mean"] == 0
    assert rows == before


def test_adding_no_progress_turns_does_not_change_other_turn_credit():
    a = compute_group_credit([row()], config())["trajectories"][0]
    b = compute_group_credit([row(phi=(0, 0, .5, 1, 1, 1))], config())["trajectories"][0]
    assert turns(a) == [.25, .25, 0]
    assert turns(b) == [0, .25, .25, 0, 0]


def test_signed_regression_uses_current_increment_not_remaining_return():
    value = compute_group_credit([row(phi=(0, 1, .5, .5), terminal=0)], config())["trajectories"][0]
    assert turns(value) == [.5, -.25, 0]


def test_task_advantage_still_broadcast_and_progress_does_not_imply_positive():
    a = row(0, phi=(0, .5, .5), terminal=0)  # R=.25
    b = row(1, phi=(0, 1, 1), terminal=1)    # R=1.5
    results = compute_group_credit([a, b], config())["trajectories"]
    # mean=.875, population std=.625; epsilon=1e-6.
    assert turns(results[0]) == pytest.approx([-.74999840000256, -.99999840000256])
    assert turns(results[1]) == pytest.approx([1.49999840000256, .99999840000256])


def test_policy_floor_preserves_process_cost_and_no_double_simple_penalty():
    a = row(0, bad=(1,), costs=(.08, .1, .03))
    value = compute_group_credit([a], config())["trajectories"][0]
    assert turns(value) == pytest.approx([.17, -1.1, -.03])


def test_zero_truncation_wins_over_all_delta_and_policy_terms():
    a = row(0, bad=(1,), costs=(.08, .1, .03))
    a.update(score=0, multiplier=0, truncation_policy="zero-trajectory-v1",
             truncation_zero_applied=True, termination_reason="hard_turn_limit")
    value = compute_group_credit([a, row(1)], config())["trajectories"][0]
    assert value["trajectory_advantage"] == 0
    assert set(value["token_advantages"]) == {0}
    assert set(value["turn_returns"]) == {0}


def test_legacy_remaining_return_path_is_unchanged():
    report = compute_group_credit([row()], CreditConfig(version="procredit-turn-v4"))
    assert report["turn_mean"] == 1.25
    assert turns(report["trajectories"][0]) == [.25, 0, -.25]
    assert "progress_credit" not in report["config"]


@pytest.mark.parametrize("extra", [
    {"db_subgoal_credit": "db_subgoal_local_v1"},
    {"post_completion_credit": "post_completion_regression_v1"},
])
def test_immediate_mode_rejects_unrequested_additive_components(extra):
    with pytest.raises(ValueError, match="separate"):
        config(**extra)

import json

import pytest

from tau2_agentic_rl.ppo_audit import audit_update, ratio_statistics


def test_live_boundary_audit_uses_current_policy_and_preserves_old(scratch_dir):
    calls = []

    def compute():
        calls.append("inference")
        return [[-0.1001, -200, -0.1999]]

    def update():
        calls.append("update")
        return "real-result"

    result, report = audit_update(
        read_old=lambda: ([[-0.1, 0, -0.2]], [[1, 0, 1]]),
        compute_current=compute,
        update=update,
        path=scratch_dir / "audit.json",
    )
    assert calls == ["inference", "update"]
    assert result == "real-result"
    assert report["ratio_mean"] == pytest.approx(1, abs=0.001)
    assert (
        report["old_log_probs_sha256_before_update"]
        == report["old_log_probs_sha256_after_update"]
    )
    assert not report["epoch_boundaries_instrumented"]


@pytest.mark.parametrize("current,expected_ratio", [(-1.0, 0.4065696597), (-0.01, 1.0941742837)])
def test_backend_logprob_difference_is_diagnostic_and_keeps_behavior_policy(
    scratch_dir, current, expected_ratio
):
    calls = []
    old = [[-0.1, 0.0]]

    def update():
        # The optimizer must still see the sampled policy, not the FSDP value.
        calls.append(list(old[0]))
        return "updated"

    result, report = audit_update(
        read_old=lambda: ([list(row) for row in old], [[1, 0]]),
        compute_current=lambda: [[current, -200.0]],
        update=update,
        path=scratch_dir / "audit.json",
        tolerance=0.005,
    )
    assert result == "updated"
    assert calls == [[-0.1, 0.0]]
    assert old == [[-0.1, 0.0]]
    assert report["ratio_mean"] == pytest.approx(expected_ratio)
    assert report["behavior_policy_source"] == "vllm_rollout"
    assert report["log_prob_abs_diff_mean"] == pytest.approx(abs(current + 0.1))
    assert report["log_prob_abs_diff_exceeds_tolerance"] is True
    assert report["status"] == "update_boundary_passed"
    assert (
        report["old_log_probs_sha256_before_update"]
        == report["old_log_probs_sha256_after_update"]
    )
    assert json.loads((scratch_dir / "audit.json").read_text()) == report


@pytest.mark.parametrize(
    "old,current,mask",
    [
        ([float("nan")], [-1.0], [1]),
        ([-0.1], [float("inf")], [1]),
        ([-0.1], [-1.0], [2]),
        ([-0.1, -0.2], [-1.0, -2.0], [1, 0]),
        ([-0.1], [-1.0, -2.0], [1]),
    ],
)
def test_invalid_probability_data_still_blocks_update(scratch_dir, old, current, mask):
    calls = []
    with pytest.raises(ValueError):
        audit_update(
            read_old=lambda: ([old], [mask]),
            compute_current=lambda: [current],
            update=lambda: calls.append("update"),
            path=scratch_dir / "audit.json",
        )
    assert not calls


def test_old_log_probs_changes_are_detected(scratch_dir):
    old = [[-0.1]]

    def update():
        old[0] = [-0.2]

    with pytest.raises(ValueError, match="changed during"):
        audit_update(
            read_old=lambda: ([list(row) for row in old], [[1]]),
            compute_current=lambda: [[-0.1]],
            update=update,
            path=scratch_dir / "audit.json",
        )


@pytest.mark.parametrize("values", [[float("nan")], [float("inf")]])
def test_nonfinite_ratios_cannot_pass(values):
    with pytest.raises(ValueError, match="non-finite"):
        ratio_statistics(values, [0], [1])

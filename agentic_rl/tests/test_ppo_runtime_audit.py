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


def test_ratio_mismatch_blocks_optimizer(scratch_dir):
    calls = []
    with pytest.raises(ValueError, match="misaligned"):
        audit_update(
            read_old=lambda: ([[-0.1]], [[1]]),
            compute_current=lambda: [[-1]],
            update=lambda: calls.append(1),
            path=scratch_dir / "audit.json",
        )
    assert not calls
    assert json.loads((scratch_dir / "audit.json").read_text())["ratio_mean"] < 1


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

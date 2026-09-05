import pytest

from tau2_agentic_rl.schemas import TrajectoryRecord
from tau2_agentic_rl.storage import TrajectoryStore


def record():
    return TrajectoryRecord(
        trajectory_id="fixture",
        task_id="1",
        split="test",
        policy_version=0,
        annotation_version="test",
        reward_version="test",
        termination_reason="user_stop",
        assistant_turns=1,
        trajectory_tokens=10,
    )


def test_store_cannot_relabel_foreign_evaluation(monkeypatch, scratch_dir):
    monkeypatch.setenv("EVALUATION_MANIFEST_ID", "new-evaluation")
    value = record()
    value.metadata["evaluation_manifest_id"] = "old-evaluation"
    with pytest.raises(ValueError, match="cannot relabel"):
        TrajectoryStore(scratch_dir).save(value)


def test_rescored_data_does_not_inherit_eval_identity_from_environment(
    monkeypatch, scratch_dir
):
    monkeypatch.setenv("EVALUATION_MANIFEST_ID", "frozen-evaluation")
    value = record()
    store = TrajectoryStore(scratch_dir, attach_evaluation_identity=False)
    store.save(value)
    assert "evaluation_manifest_id" not in next(store.records()).metadata

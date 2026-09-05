import json

import pytest

from scripts.summarize_evaluation import summarize
from tau2_agentic_rl.evaluation import (
    evaluation_coverage,
    evaluation_lock,
    initialize_evaluation,
)


def setup_evaluation(scratch_dir, count=20):
    identity = {
        "model_files": {"model.safetensors": "model-A"},
        "task_ids": [str(i) for i in range(count)],
        "samples_per_task": 4,
        "record_split": "test" if count == 20 else "internal_dev",
        "split": "official_test" if count == 20 else "internal_dev",
    }
    root = scratch_dir / "evaluation"
    manifest = initialize_evaluation(root, identity, resume=False)
    records = root / "trajectories"
    records.mkdir()
    return root, records, manifest


def write_sample(records, manifest, task, slot, *, failed=False, name=None):
    row = {
        "trajectory_id": name or f"{task}-{slot}",
        "task_id": str(task),
        "split": manifest["identity"]["record_split"],
        "metadata": {
            "evaluation_manifest_id": manifest["manifest_id"],
            "evaluation_sample_index": slot,
        },
        "termination_reason": "infrastructure_error" if failed else "user_stop",
        "official_scores": None if failed else {"reward": float(slot == 0)},
        "custom_reward": None if failed else {"strict_success": float(slot == 0)},
    }
    path = records / f"{row['trajectory_id']}.json"
    path.write_text(json.dumps(row), encoding="utf-8")
    return path


def test_final_test_requires_all_twenty_tasks_and_eighty_valid_slots(scratch_dir):
    _, records, manifest = setup_evaluation(scratch_dir)
    for task in range(20):
        for slot in range(4):
            if (task, slot) != (19, 3):
                write_sample(records, manifest, task, slot)
    write_sample(records, manifest, 19, 3, failed=True, name="failed-api-attempt")
    with pytest.raises(ValueError, match="no final metrics"):
        summarize(records)
    partial = summarize(records, allow_incomplete=True)
    assert "aggregate" not in partial and "per_task" not in partial
    assert partial["missing_slots"] == [{"task_id": "19", "sample_index": 3}]
    write_sample(records, manifest, 19, 3, name="refill")
    result = summarize(records)
    assert result["valid_samples"] == 80
    assert result["infrastructure_failures"] == 1
    assert result["tasks"] == 20
    assert result["aggregate"]["official_pass1"] == 0.25
    assert result["aggregate"]["official_pass4"] == 1.0


def test_missing_whole_task_is_not_dropped_from_macro_average(scratch_dir):
    _, records, manifest = setup_evaluation(scratch_dir, count=2)
    for slot in range(4):
        write_sample(records, manifest, 0, slot)
    coverage = evaluation_coverage(records, manifest)
    assert len(coverage["missing_slots"]) == 4
    with pytest.raises(ValueError, match="incomplete"):
        summarize(records)


def test_nonempty_tag_and_changed_model_cannot_mix_samples(scratch_dir):
    root, _, manifest = setup_evaluation(scratch_dir, count=1)
    with pytest.raises(FileExistsError):
        initialize_evaluation(root, manifest["identity"], resume=False)
    assert initialize_evaluation(root, manifest["identity"], resume=True) == manifest
    changed = {**manifest["identity"], "model_files": {"model.safetensors": "model-B"}}
    with pytest.raises(ValueError, match="identity changed"):
        initialize_evaluation(root, changed, resume=True)


def test_foreign_or_duplicate_valid_records_are_rejected(scratch_dir):
    _, records, manifest = setup_evaluation(scratch_dir, count=1)
    write_sample(records, manifest, 0, 0)
    write_sample(records, manifest, 0, 0, name="duplicate-slot")
    with pytest.raises(ValueError, match="duplicate valid"):
        evaluation_coverage(records, manifest)


def test_foreign_manifest_is_rejected(scratch_dir):
    _, records, manifest = setup_evaluation(scratch_dir, count=1)
    foreign = {**manifest, "manifest_id": "different-model"}
    write_sample(records, foreign, 0, 0)
    with pytest.raises(ValueError, match="foreign"):
        evaluation_coverage(records, manifest)


def test_concurrent_refills_cannot_claim_the_same_slots(scratch_dir):
    with evaluation_lock(scratch_dir):
        with pytest.raises(FileExistsError):
            with evaluation_lock(scratch_dir):
                pytest.fail("second process must not acquire the lock")
    assert not (scratch_dir / "evaluation.lock").exists()

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_review_boundaries import scoring_failure_record

from scripts import evaluate_airline


@pytest.mark.parametrize("scoring_only", [False, True])
def test_launcher_refills_only_missing_slots_and_resume_is_identity_bound(
    monkeypatch, scratch_dir, scoring_only
):
    source = Path(__file__).parents[1]
    for directory in ("configs", "data/annotations", "data/splits"):
        shutil.copytree(source / directory, scratch_dir / directory)
    (scratch_dir / "scripts").mkdir()
    model = scratch_dir / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fixture-not-real-weights")
    monkeypatch.setattr(
        evaluate_airline, "__file__", str(scratch_dir / "scripts/evaluate_airline.py")
    )
    monkeypatch.setattr(evaluate_airline, "_require_exact_checkout", lambda *args: None)
    for key in (
        "DEEPSEEK_USER_MODEL",
        "DEEPSEEK_JUDGE_MODEL",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
    ):
        monkeypatch.setenv(key, "fixture")
    # Restore all environment writes that the real launcher makes.
    for key in (
        "AGENTIC_RL_CONFIG",
        "TRAJECTORY_OUTPUT_DIR",
        "JUDGE_CACHE_DIR",
        "USER_CACHE_DIR",
        "AGENTIC_RL_PROJECT_ROOT",
        "MERGED_SFT_MODEL",
        "PYTHONPATH",
        "EVALUATION_MANIFEST_ID",
    ):
        monkeypatch.setenv(key, "")
    argv = [
        "evaluate",
        "--tau2-root",
        str(scratch_dir),
        "--verl-root",
        str(scratch_dir),
        "--model-path",
        str(model),
        "--tag",
        "test-run",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    root = scratch_dir / "outputs/evaluations/test-run"
    batches = []

    def write_parquet(rows, path):
        assert path == root / "pending_samples.parquet"
        batches.append(rows)

    def launch(command, **kwargs):
        assert "actor_rollout_ref.rollout.val_kwargs.n=1" in command
        assert "trainer.resume_mode=disable" in command
        assert (root / "runtime_config.yaml").exists()
        manifest = json.loads(
            (root / "evaluation_manifest.json").read_text(encoding="utf-8")
        )
        records = root / "trajectories"
        records.mkdir(exist_ok=True)
        for index, row in enumerate(batches[-1]):
            failed = len(batches) == 1 and index == 0
            metadata = row["extra_info"]
            record = {
                "trajectory_id": f"{len(batches)}-{index}",
                "task_id": metadata["task_id"],
                "split": metadata["split"],
                "metadata": {
                    "evaluation_sample_index": metadata["evaluation_sample_index"],
                    "evaluation_manifest_id": manifest["manifest_id"],
                },
                "termination_reason": "infrastructure_error" if failed else "user_stop",
                "official_scores": None if failed else {"reward": 0.0},
                "custom_reward": None if failed else {"strict_success": 0.0},
            }
            if failed and scoring_only:
                pending = scoring_failure_record().model_dump()
                pending.update(
                    {key: record[key] for key in ("trajectory_id", "task_id", "split")}
                )
                pending["metadata"].update(record["metadata"])
                record = pending
            (records / f"{record['trajectory_id']}.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
        return SimpleNamespace(returncode=1 if len(batches) == 1 else 0)

    monkeypatch.setattr(evaluate_airline, "_write_parquet", write_parquet)
    monkeypatch.setattr(evaluate_airline.subprocess, "run", launch)

    async def judge(_self, **inputs):
        from tau2_agentic_rl.schemas import JudgeResult

        return JudgeResult(), "raw", "prompt", "cache"

    monkeypatch.setattr(evaluate_airline.DeepSeekJudge, "evaluate", judge)
    evaluate_airline.main()
    assert [len(rows) for rows in batches] == ([80] if scoring_only else [80, 1])
    if not scoring_only:
        assert batches[1][0] == batches[0][0]
    result = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert result["valid_samples"] == 80
    assert result["aggregate"]["official_pass1"] == (0.0125 if scoring_only else 0)
    assert result["aggregate"]["official_pass4"] == 0
    with pytest.raises(FileExistsError):
        evaluate_airline.main()
    monkeypatch.setattr(sys, "argv", argv + ["--resume"])
    evaluate_airline.main()
    assert len(batches) == (1 if scoring_only else 2)
    (model / "model.safetensors").write_bytes(b"different-model")
    with pytest.raises(ValueError, match="identity changed"):
        evaluate_airline.main()

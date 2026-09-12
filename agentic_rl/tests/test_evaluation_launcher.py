import asyncio
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_review_boundaries import scoring_failure_record

from scripts import evaluate_airline


@pytest.mark.parametrize(
    "flags,expected",
    [([], False), (["--reward-judge"], True), (["--no-reward-judge"], False)],
)
def test_agent_reward_judge_cli_switch(flags, expected):
    assert evaluate_airline.parse_args(flags).reward_judge is expected


@pytest.mark.parametrize("scoring_only", [False, True])
@pytest.mark.parametrize("reward_judge", [False, True])
def test_launcher_refills_only_missing_slots_and_resume_is_identity_bound(
    monkeypatch, scratch_dir, scoring_only, reward_judge
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
        "DEEPSEEK_USER_SIM_JUDGE_MODEL",
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
        "USER_SIM_JUDGE_CACHE_DIR",
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
    if reward_judge:
        argv.append("--reward-judge")
    else:
        monkeypatch.delenv("DEEPSEEK_JUDGE_MODEL", raising=False)
    monkeypatch.setattr(sys, "argv", argv)
    root = scratch_dir / "outputs/evaluations/test-run"
    batches = []
    pending_count = 7 if scoring_only else 1
    active_scoring, peak_scoring, scored = 0, 0, 0

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
            failed = len(batches) == 1 and index < pending_count
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
            if record.get("official_scores") is not None:
                from test_user_simulation import attach_valid_screen

                attach_valid_screen(record)
            (records / f"{record['trajectory_id']}.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
        return SimpleNamespace(returncode=1 if len(batches) == 1 else 0)

    monkeypatch.setattr(evaluate_airline, "_write_parquet", write_parquet)
    monkeypatch.setattr(evaluate_airline.subprocess, "run", launch)

    async def judge(_self, **inputs):
        from tau2_agentic_rl.schemas import JudgeResult

        nonlocal active_scoring, peak_scoring, scored
        assert reward_judge, "official-only pass must never invoke Agent Judge"
        active_scoring += 1
        peak_scoring = max(peak_scoring, active_scoring)
        try:
            await asyncio.sleep(0)
            scored += 1
            return JudgeResult(), "raw", "prompt", "cache"
        finally:
            active_scoring -= 1

    monkeypatch.setattr(evaluate_airline.DeepSeekJudge, "evaluate", judge)
    evaluate_airline.main()
    assert [len(rows) for rows in batches] == ([80] if scoring_only else [80, 1])
    if not scoring_only:
        assert batches[1][0] == batches[0][0]
    result = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert result["valid_samples"] == 80
    assert result["aggregate"]["official_pass1"] == pytest.approx(
        pending_count / 80 if scoring_only else 0
    )
    assert result["aggregate"]["official_pass4"] == (0.05 if scoring_only else 0)
    assert peak_scoring == (4 if scoring_only and reward_judge else 0)
    assert active_scoring == 0
    assert scored == (pending_count if scoring_only and reward_judge else 0)
    assert ("custom_strict_pass1" in result["aggregate"]) == reward_judge
    with pytest.raises(FileExistsError):
        evaluate_airline.main()
    monkeypatch.setattr(sys, "argv", argv + ["--resume"])
    evaluate_airline.main()
    assert len(batches) == (1 if scoring_only else 2)
    assert scored == (pending_count if scoring_only and reward_judge else 0)
    (model / "model.safetensors").write_bytes(b"different-model")
    with pytest.raises(ValueError, match="identity changed"):
        evaluate_airline.main()


def test_launcher_rejects_wrong_twenty_test_ids_before_model_hashing_or_launch(
    monkeypatch,
    scratch_dir,
):
    source = Path(__file__).parents[1]
    for directory in ("configs", "data/splits"):
        shutil.copytree(source / directory, scratch_dir / directory)
    split_path = scratch_dir / "data/splits/airline_internal_dev.v1.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["official_test"] = [str(task) for task in range(20)]
    split_path.write_text(json.dumps(split), encoding="utf-8")
    model = scratch_dir / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fixture")
    monkeypatch.setattr(
        evaluate_airline, "__file__", str(scratch_dir / "scripts/evaluate_airline.py")
    )
    monkeypatch.setattr(evaluate_airline, "_require_exact_checkout", lambda *args: None)
    for key in (
        "DEEPSEEK_USER_MODEL",
        "DEEPSEEK_JUDGE_MODEL",
        "DEEPSEEK_USER_SIM_JUDGE_MODEL",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
    ):
        monkeypatch.setenv(key, "fixture")
    # Restore all launcher environment writes after the preflight test.
    for key in (
        "AGENTIC_RL_CONFIG",
        "TRAJECTORY_OUTPUT_DIR",
        "JUDGE_CACHE_DIR",
        "USER_CACHE_DIR",
        "USER_SIM_JUDGE_CACHE_DIR",
        "AGENTIC_RL_PROJECT_ROOT",
        "MERGED_SFT_MODEL",
    ):
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(
        evaluate_airline,
        "fingerprint_directory",
        lambda *args: pytest.fail("must reject before hashing"),
    )
    monkeypatch.setattr(
        evaluate_airline.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("must not launch"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--tau2-root",
            str(scratch_dir),
            "--verl-root",
            str(scratch_dir),
            "--model-path",
            str(model),
            "--tag",
            "must-not-create",
            "--dry-run",
        ],
    )
    with pytest.raises(ValueError, match="20 official test IDs"):
        evaluate_airline.main()
    assert not (scratch_dir / "outputs/evaluations/must-not-create").exists()


@pytest.mark.parametrize("quality_api_fails", [False, True])
def test_user_filter_refills_invalid_slot_but_never_rerolls_pending_check(
    monkeypatch, scratch_dir, quality_api_fails
):
    from test_user_simulation import attach_valid_screen, verdict

    from scripts.summarize_evaluation import summarize
    from tau2_agentic_rl.schemas import OfficialScores
    from tau2_agentic_rl.user_simulation import UserSimulationVerdict, replacement_seed

    source = Path(__file__).parents[1]
    for directory in ("configs", "data/annotations", "data/splits"):
        shutil.copytree(source / directory, scratch_dir / directory)
    (scratch_dir / "scripts").mkdir()
    model = scratch_dir / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fixture")
    monkeypatch.setattr(
        evaluate_airline, "__file__", str(scratch_dir / "scripts/evaluate_airline.py")
    )
    monkeypatch.setattr(evaluate_airline, "_require_exact_checkout", lambda *a: None)
    for key in (
        "DEEPSEEK_USER_MODEL",
        "DEEPSEEK_USER_SIM_JUDGE_MODEL",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
    ):
        monkeypatch.setenv(key, "fixture")
    monkeypatch.delenv("DEEPSEEK_JUDGE_MODEL", raising=False)
    for key in (
        "AGENTIC_RL_CONFIG",
        "TRAJECTORY_OUTPUT_DIR",
        "JUDGE_CACHE_DIR",
        "USER_CACHE_DIR",
        "USER_SIM_JUDGE_CACHE_DIR",
        "AGENTIC_RL_PROJECT_ROOT",
        "MERGED_SFT_MODEL",
        "PYTHONPATH",
        "EVALUATION_MANIFEST_ID",
    ):
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--tau2-root",
            str(scratch_dir),
            "--verl-root",
            str(scratch_dir),
            "--model-path",
            str(model),
            "--tag",
            "screened",
        ],
    )
    root = scratch_dir / "outputs/evaluations/screened"
    batches = []
    checked = []

    class QualityJudge:
        async def evaluate(self, **inputs):
            checked.append(inputs)
            if quality_api_fails:
                raise RuntimeError("quality API unavailable")
            return UserSimulationVerdict.model_validate(verdict()), "raw", "hash", "key"

    monkeypatch.setattr(
        evaluate_airline, "build_user_sim_judge", lambda *a: QualityJudge()
    )
    monkeypatch.setattr(
        evaluate_airline,
        "DeepSeekJudge",
        lambda *a: pytest.fail("Agent reward Judge must not even be constructed"),
    )
    monkeypatch.setattr(
        evaluate_airline, "_write_parquet", lambda rows, path: batches.append(rows)
    )

    def launch(*args, **kwargs):
        manifest = json.loads(
            (root / "evaluation_manifest.json").read_text(encoding="utf-8")
        )
        for index, row in enumerate(batches[-1]):
            extra = row["extra_info"]
            record = scoring_failure_record()
            record.trajectory_id = f"{len(batches)}-{index}"
            record.task_id, record.split = extra["task_id"], extra["split"]
            record.environment_seed = extra["environment_seed"]
            record.official_scores = OfficialScores(reward=0)
            record.metadata.update(
                evaluation_manifest_id=manifest["manifest_id"],
                evaluation_sample_index=extra["evaluation_sample_index"],
                user_sim_attempt=extra["user_sim_attempt"],
                reward_judge_enabled=False,
                failure_phase=None,
            )
            data = attach_valid_screen(
                record.model_dump(), valid=not (len(batches) == 1 and index == 0)
            )
            if len(batches) == 1 and index == 1:
                data["user_sim_result"] = None
                data["metadata"]["failure_phase"] = "user_sim_judge"
            (root / "trajectories" / f"{record.trajectory_id}.json").write_text(
                json.dumps(data), encoding="utf-8"
            )
        return SimpleNamespace(returncode=1 if len(batches) == 1 else 0)

    monkeypatch.setattr(evaluate_airline.subprocess, "run", launch)
    if quality_api_fails:
        with pytest.raises(ValueError, match="incomplete evaluation"):
            evaluate_airline.main()
    else:
        evaluate_airline.main()
    assert [len(batch) for batch in batches] == [80, 1]
    first, replacement = batches[0][0]["extra_info"], batches[1][0]["extra_info"]
    assert replacement["task_id"] == first["task_id"]
    assert replacement["evaluation_sample_index"] == first["evaluation_sample_index"]
    assert replacement["user_sim_attempt"] == 1
    assert replacement["environment_seed"] == replacement_seed(
        first["environment_seed"], 1
    )
    report = summarize(root / "trajectories", allow_incomplete=True)
    assert report["user_sim_rejections"] == 1
    assert report["valid_samples"] == (79 if quality_api_fails else 80)
    assert not report["missing_slots"]
    assert len(report["user_sim_pending_slots"]) == int(quality_api_fails)
    assert checked and all(set(call) == {"inputs"} for call in checked)
    if not quality_api_fails:
        assert report["aggregate"] == {"official_pass1": 0, "official_pass4": 0}

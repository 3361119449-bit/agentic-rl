import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import train_airline_grpo as launcher
from tau2_agentic_rl.config import load_yaml
from tau2_agentic_rl.rl_resume import read_resume_identity
from tau2_agentic_rl.versions import sha256_file

ROOT = Path(__file__).parents[1]


def _write_tasks(path: Path, task_ids: list[str], split: str = "train") -> None:
    rows = [
        {
            "data_source": "tau2_airline",
            "extra_info": {
                "task_id": task_id,
                "split": split,
                "environment_seed": index,
            },
        }
        for index, task_id in enumerate(task_ids)
    ]
    pq.write_table(
        pa.Table.from_pylist(rows),
        path,
    )


def _task_ids(path: Path) -> list[str]:
    return [
        str(row["extra_info"]["task_id"]) for row in pq.read_table(path).to_pylist()
    ]


def test_filtered_training_parquet_is_deterministic_and_preserves_source(scratch_dir):
    source = scratch_dir / "airline_official_train.parquet"
    _write_tasks(source, ["6", "7", "8"])
    original = source.read_bytes()

    planned = launcher.prepare_filtered_training_parquet(
        source, ["7", "7"], materialize=False
    )
    assert planned.parent == source.parent / "derived"
    assert not planned.exists()

    filtered = launcher.prepare_filtered_training_parquet(
        source, ["7"], materialize=True
    )
    assert filtered == planned
    assert _task_ids(filtered) == ["6", "8"]
    assert source.read_bytes() == original
    assert (
        launcher.prepare_filtered_training_parquet(source, ["7"], materialize=True)
        == filtered
    )


def test_filtered_training_parquet_rejects_absent_or_invalid_task(scratch_dir):
    source = scratch_dir / "airline_rl_train.parquet"
    _write_tasks(source, ["0", "1"])

    with pytest.raises(ValueError, match="absent from selected training data: 7"):
        launcher.prepare_filtered_training_parquet(source, ["7"], materialize=False)
    with pytest.raises(ValueError, match="nonnegative decimal"):
        launcher.prepare_filtered_training_parquet(
            source, ["task-1"], materialize=False
        )


def test_real_launcher_excludes_only_training_rows_and_records_identity(
    monkeypatch, scratch_dir
):
    for directory in ("configs", "data/annotations"):
        source = ROOT / directory
        target = scratch_dir / directory
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
    parquet = scratch_dir / "data/parquet"
    parquet.mkdir()
    _write_tasks(parquet / "airline_official_train.parquet", ["6", "7", "8"])
    _write_tasks(parquet / "airline_rl_train.parquet", ["6", "8"])
    _write_tasks(parquet / "airline_internal_dev.parquet", ["7"], "internal_dev")
    base = scratch_dir / "base"
    base.mkdir()
    for name in ("config.json", "tokenizer_config.json", "model.safetensors"):
        (base / name).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        launcher, "__file__", str(scratch_dir / "scripts/train_airline_grpo.py")
    )
    monkeypatch.setattr(launcher, "_require_exact_checkout", lambda *args: None)
    monkeypatch.setenv("MERGED_SFT_MODEL", str(base))
    for key in (
        "DEEPSEEK_USER_MODEL",
        "DEEPSEEK_JUDGE_MODEL",
        "DEEPSEEK_USER_SIM_JUDGE_MODEL",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
    ):
        monkeypatch.setenv(key, "fixture")
    for key in (
        "TRAJECTORY_OUTPUT_DIR",
        "CHECKPOINT_OUTPUT_DIR",
        "JUDGE_CACHE_DIR",
        "USER_CACHE_DIR",
        "USER_SIM_JUDGE_CACHE_DIR",
        "METRICS_OUTPUT_DIR",
        "REPORTS_OUTPUT_DIR",
        "AGENTIC_RL_CONFIG",
        "AGENTIC_RL_PROJECT_ROOT",
        "PYTHONPATH",
    ):
        monkeypatch.setenv(key, "")
    launches = []
    monkeypatch.setattr(
        launcher.subprocess, "run", lambda command, **kw: launches.append(command)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--tau2-root",
            str(scratch_dir),
            "--verl-root",
            str(scratch_dir),
            "--stage",
            "full_train",
            "--exclude-task-id",
            "7",
            "--run-name",
            "full-train-without-7",
        ],
    )

    launcher.main()

    assert len(launches) == 1
    train_override = next(
        item for item in launches[0] if item.startswith("data.train_files=")
    )
    filtered = Path(train_override.split("=", 1)[1])
    assert _task_ids(filtered) == ["6", "8"]
    assert _task_ids(parquet / "airline_official_train.parquet") == ["6", "7", "8"]
    assert _task_ids(parquet / "airline_internal_dev.parquet") == ["7"]

    run = scratch_dir / "outputs/runs/full-train-without-7"
    runtime = load_yaml(run / "runtime_config.yaml")
    identity = read_resume_identity(run)
    expected = {"excluded_train_task_ids": ["7"]}
    assert runtime["data_selection"] == expected
    assert identity["runtime_config"]["data_selection"] == expected
    assert identity["files"]["data.train_files"] == sha256_file(filtered)

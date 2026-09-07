import ast
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import train_airline_grpo as launcher
from tau2_agentic_rl.config import load_yaml
from tau2_agentic_rl.rl_resume import (
    MANIFEST,
    build_resume_identity,
    read_resume_identity,
    require_same_identity,
    save_resume_identity,
    snapshot_resume_identity,
)
from tau2_agentic_rl.training_config import effective_project_config

ROOT = Path(__file__).parents[1]


def identity_fixture(root, *, seed=42, epochs=15, stage="internal_dev", extra=None):
    project = load_yaml(ROOT / "configs/rl/airline_grpo_v1.yaml")
    train = root / "train.parquet"
    val = root / "dev.parquet"
    for path in (train, val):
        if not path.exists():
            path.write_bytes(b"fingerprint fixture, not read as Parquet")
    command = launcher.build_command(
        project_root=ROOT,
        model_path="/same-base",
        train_file=train,
        val_file=val,
        total_epochs=epochs,
        seed=seed,
        extra=extra or [],
        project_config=project,
    )
    return build_resume_identity(ROOT, project, command, stage=stage)


@pytest.mark.parametrize(
    "change",
    ["stage", "seed", "epochs", "train", "dev", "extra", "learning_rate", "annotation"],
)
def test_changed_training_identity_is_rejected_even_in_new_destination(
    scratch_dir, change
):
    original = identity_fixture(scratch_dir)
    checkpoint = scratch_dir / "checkpoint"
    save_resume_identity(checkpoint, original)
    options = {}
    if change in ("train", "dev"):
        (scratch_dir / f"{change}.parquet").write_bytes(b"changed data at SAME path")
    elif change == "extra":
        options["extra"] = ["trainer.total_training_steps=7"]
    elif change in ("stage", "seed", "epochs"):
        options[change] = {"stage": "full_train", "seed": 43, "epochs": 16}[change]
    changed = identity_fixture(scratch_dir, **options)
    if change == "learning_rate":
        changed["runtime_config"]["optimizer"]["lr"] *= 2
    elif change == "annotation":
        changed["files"]["annotations.required_actions"] = "changed"
    destination = scratch_dir / "new-run"
    with pytest.raises(ValueError, match="resume identity changed"):
        launcher.prepare_run_directory(
            destination, checkpoint, "/unused-base", {}, resume_identity=changed
        )
    assert not destination.exists()
    assert read_resume_identity(checkpoint) == original


def test_relocation_changes_paths_but_not_identity(scratch_dir):
    original = identity_fixture(scratch_dir)
    # Use actual YAML paths for content hashing, not the normalized snapshot.
    project = load_yaml(ROOT / "configs/rl/airline_grpo_v1.yaml")
    project["outputs"] = {"checkpoints": "/relocated/checkpoints"}
    project["model"]["sft_input"] = "/relocated/base"
    relocated = scratch_dir / "relocated"
    relocated.mkdir()
    for filename in ("train.parquet", "dev.parquet"):
        shutil.copyfile(scratch_dir / filename, relocated / filename)
    command = launcher.build_command(
        project_root=ROOT,
        model_path="/relocated/base",
        train_file=relocated / "train.parquet",
        val_file=relocated / "dev.parquet",
        total_epochs=15,
        extra=[],
        run_name="new-name",
        run_root=relocated,
        project_config=project,
    )
    command = [
        "trainer.resume_mode=resume_path"
        if item == "trainer.resume_mode=disable"
        else item
        for item in command
    ] + ["trainer.resume_from_path=/original/checkpoints/global_step_10"]
    assert (
        build_resume_identity(ROOT, project, command, stage="internal_dev") == original
    )


def test_legacy_checkpoint_cannot_borrow_a_new_run_manifest(scratch_dir):
    identity = identity_fixture(scratch_dir)
    save_resume_identity(scratch_dir, identity)
    checkpoint = scratch_dir / "checkpoints/global_step_10"
    checkpoint.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="legacy or incomplete"):
        launcher.prepare_run_directory(
            scratch_dir, checkpoint, "/unused-base", {}, resume_identity=identity
        )
    assert not (checkpoint / MANIFEST).exists()


@pytest.mark.parametrize(
    "override",
    [
        "data.seed=43",
        "actor_rollout_ref.actor.data_loader_seed=43",
        "actor_rollout_ref.rollout.seed=43",
        "trainer.total_epochs=2",
    ],
)
def test_extra_cannot_override_dedicated_seed_or_epochs(override):
    with pytest.raises(ValueError, match="dedicated"):
        effective_project_config(
            load_yaml(ROOT / "configs/rl/airline_grpo_v1.yaml"), [override]
        )


def test_actual_checkpoint_save_snapshots_identity_after_save(scratch_dir):
    source = ROOT / "src/tau2_agentic_rl/verl_capped_trainer.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    trainer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CappedPPOTrainerSync"
    )
    method = next(
        node
        for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "_save_checkpoint"
    )
    wrapper = ast.parse("class UnderTest(Parent):\n    pass\n")
    wrapper.body[0].body = [method]
    ast.fix_missing_locations(wrapper)
    identity = identity_fixture(scratch_dir)
    save_resume_identity(scratch_dir, identity)
    checkpoint = scratch_dir / "checkpoints/global_step_10"

    class Parent:
        def _save_checkpoint(self):
            checkpoint.mkdir(parents=True)
            assert not (checkpoint / MANIFEST).exists()

        def _write_step_counters(self, clock, path):
            assert checkpoint.exists()
            path.write_text(json.dumps(clock), encoding="utf-8")

    scope = {
        "Parent": Parent,
        "Path": Path,
        "snapshot_resume_identity": snapshot_resume_identity,
    }
    exec(compile(wrapper, str(source), "exec"), scope)
    instance = scope["UnderTest"]()
    instance.config = SimpleNamespace(
        trainer=SimpleNamespace(default_local_dir=str(checkpoint.parent))
    )
    instance.global_steps = 10
    instance.step_clock = {"optimizer_step": 10}
    instance._save_checkpoint()
    require_same_identity(checkpoint, identity)


def test_real_launcher_fresh_resume_and_changed_cli_fail_before_launch(
    monkeypatch, scratch_dir
):
    # Run the production launcher; only external checkout checks/GPU subprocess are stubbed.
    for directory in ("configs", "data/annotations"):
        shutil.copytree(ROOT / directory, scratch_dir / directory)
    parquet = scratch_dir / "data/parquet"
    parquet.mkdir()
    for name in ("airline_rl_train", "airline_official_train", "airline_internal_dev"):
        (parquet / f"{name}.parquet").write_bytes(name.encode())
    base = scratch_dir / "base"
    base.mkdir()
    for name in ("config.json", "tokenizer_config.json", "model.safetensors"):
        (base / name).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        launcher, "__file__", str(scratch_dir / "scripts/train_airline_grpo.py")
    )
    monkeypatch.setattr(launcher, "_require_exact_checkout", lambda *args: None)
    for key in (
        "DEEPSEEK_USER_MODEL",
        "DEEPSEEK_JUDGE_MODEL",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
    ):
        monkeypatch.setenv(key, "fixture")
    monkeypatch.setenv("MERGED_SFT_MODEL", str(base))
    for key in (
        "TRAJECTORY_OUTPUT_DIR",
        "CHECKPOINT_OUTPUT_DIR",
        "JUDGE_CACHE_DIR",
        "USER_CACHE_DIR",
        "METRICS_OUTPUT_DIR",
        "REPORTS_OUTPUT_DIR",
        "AGENTIC_RL_CONFIG",
        "AGENTIC_RL_PROJECT_ROOT",
        "PYTHONPATH",
    ):
        monkeypatch.setenv(key, "")
    argv = [
        "train",
        "--tau2-root",
        str(scratch_dir),
        "--verl-root",
        str(scratch_dir),
        "--run-name",
        "run",
    ]
    launches = []
    monkeypatch.setattr(
        launcher.subprocess, "run", lambda command, **kw: launches.append(command)
    )
    monkeypatch.setattr(sys, "argv", argv)
    launcher.main()
    run = scratch_dir / "outputs/runs/run"
    identity = read_resume_identity(run)
    assert identity["training_overrides"]["data.seed"] == "42"
    assert identity["training_overrides"]["trainer.total_epochs"] == "15"
    checkpoint = run / "checkpoints/global_step_10"
    (checkpoint / "actor").mkdir(parents=True)
    for name in ("model", "optim", "extra_state"):
        (checkpoint / "actor" / f"{name}_world_size_1_rank_0.pt").touch()
    (checkpoint / "data.pt").touch()
    (checkpoint / "step_counters.json").write_text(
        json.dumps(
            {
                "attempt_step": 10,
                "optimizer_step": 10,
                "consecutive_skips": 0,
            }
        ),
        encoding="utf-8",
    )
    snapshot_resume_identity(run, checkpoint)
    resume_argv = argv + ["--resume-from-path", str(checkpoint)]
    monkeypatch.setattr(sys, "argv", resume_argv)
    launcher.main()
    before = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    for flags in (
        ["--seed", "43"],
        ["--epochs", "16"],
        ["--stage", "full_train"],
        ["--extra", "trainer.total_training_steps=7"],
        ["--run-name", "new", "--extra", "actor_rollout_ref.actor.optim.lr=0.000002"],
    ):
        monkeypatch.setattr(sys, "argv", resume_argv + flags)
        with pytest.raises(ValueError, match="resume identity changed"):
            launcher.main()
    assert launches and len(launches) == 2
    assert before == {
        p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()
    }
    assert not (run.parent / "new").exists()
    # Same identity may continue to a new empty directory.
    monkeypatch.setattr(sys, "argv", resume_argv + ["--run-name", "relocated"])
    launcher.main()
    assert len(launches) == 3
    assert read_resume_identity(run.parent / "relocated") == identity

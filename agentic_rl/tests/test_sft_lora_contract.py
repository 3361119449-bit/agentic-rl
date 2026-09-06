import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tau2_agentic_rl import sft_contract as contract


def load_launcher():
    path = (
        Path(__file__).parents[2] / "training/qwen3_4b_sft/train_qwen3_4b_verl_sft.py"
    )
    spec = importlib.util.spec_from_file_location("sft_lora_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(monkeypatch, directory, *flags):
    module = load_launcher()
    monkeypatch.setattr(sys, "argv", ["sft", "--output-dir", str(directory), *flags])
    args = module.parse_args()
    return module, args


def test_default_is_lora_and_absent_validation_disables_testing(
    monkeypatch, scratch_dir
):
    module, args = arguments(monkeypatch, scratch_dir / "out")
    command = module.build_training_command(args, scratch_dir / "train.parquet", None)
    assert args.lora_rank == 64 and args.global_batch_size == 8
    assert "model.lora_rank=64" in command
    assert "optim.lr=0.0001" in command
    assert "trainer.test_freq=-1" in command
    assert "data.val_files=null" in command
    assert "engine.seed=42" in command
    assert "checkpoint.save_contents=[model,optimizer,extra]" in command
    assert str(Path(module.__file__).with_name("lora_sft_runtime.py")) in command
    assert not args.output_dir.exists()  # Building/dry-running must not occupy a run.


def test_validation_enables_epoch_tests(monkeypatch, scratch_dir):
    module, args = arguments(monkeypatch, scratch_dir)
    command = module.build_training_command(
        args, scratch_dir / "train", scratch_dir / "val"
    )
    assert "trainer.test_freq=after_each_epoch" in command


@pytest.mark.parametrize(
    "flags",
    [
        ["--lora-rank", "0"],
        ["--lora-alpha", "0"],
        ["--epochs", "0"],
        ["--learning-rate", "nan"],
        ["--seed", "-1"],
        ["--resume-mode", "resume_path"],
        ["--resume-from-path", "global_step_1"],
        ["--extra-config", "model.path=another-base"],
        ["--extra-config", "data.train_files=another-data"],
        ["--extra-config", "trainer.resume_mode=auto"],
        ["--extra-config", "trainer.default_local_dir=another-run"],
        ["--extra-config", "checkpoint.load_contents=[model]"],
    ],
)
def test_reject_invalid_or_untracked_training_settings(monkeypatch, scratch_dir, flags):
    module, args = arguments(monkeypatch, scratch_dir, *flags)
    with pytest.raises(ValueError):
        module.build_training_command(args, scratch_dir / "train", None)


def make_identity():
    return {
        "schema_version": 1,
        "config": {
            "num_gpus": 2,
            "ulysses_size": 1,
            "lora_rank": 64,
            "lora_alpha": 128,
            "epochs": 2,
            "global_batch_size": 8,
            "learning_rate": 1e-4,
            "seed": 42,
        },
        "data": {
            "train_rows": 32,
            "source_sha256": "original",
            "train_sha256": "parquet",
        },
        "base_files": {"model.safetensors": "base"},
        "runtime": {"verl_commit": contract.VERL_COMMIT},
    }


def make_checkpoint(directory, identity=None, step=4):
    identity = identity or make_identity()
    path = directory / f"global_step_{step}"
    path.mkdir(parents=True)
    for name in contract.checkpoint_files(identity):
        item = path / name
        item.parent.mkdir(parents=True, exist_ok=True)
        if name != contract.RUN_MANIFEST:
            item.write_bytes(b"fixture-only-not-torch-state")
    (path / "lora_train_meta.json").write_text(
        json.dumps(
            {
                "r": 64,
                "lora_alpha": 128,
                "task_type": "CAUSAL_LM",
            }
        ),
        encoding="utf-8",
    )
    contract.publish_checkpoint(path, identity, step)
    return path, identity


def test_complete_checkpoint_resumes_and_can_move(scratch_dir):
    path, identity = make_checkpoint(scratch_dir)
    assert contract.validate_resume(path, identity) == (path.resolve(), 4)
    moved = scratch_dir / "relocated" / path.name
    moved.parent.mkdir()
    path.rename(moved)
    assert contract.validate_resume(moved, identity)[1] == 4


@pytest.mark.parametrize(
    "group,key,value",
    [
        ("config", "lora_rank", 32),
        ("config", "lora_alpha", 64),
        ("config", "global_batch_size", 16),
        ("config", "learning_rate", 2e-5),
        ("config", "seed", 99),
        ("config", "num_gpus", 1),
        ("data", "source_sha256", "changed"),
        ("data", "train_sha256", "changed"),
        ("base_files", "model.safetensors", "changed"),
    ],
)
def test_changed_run_cannot_resume(scratch_dir, group, key, value):
    path, identity = make_checkpoint(scratch_dir)
    changed = copy.deepcopy(identity)
    changed[group][key] = value
    with pytest.raises(ValueError, match="identity mismatch"):
        contract.validate_resume(path, changed)


@pytest.mark.parametrize(
    "filename",
    [
        contract.COMPLETE_MANIFEST,
        "data_1.pt",
        "optim_world_size_2_rank_1.pt",
        "extra_state_world_size_2_rank_0.pt",
        "model_world_size_2_rank_1.pt",
    ],
)
def test_partial_checkpoint_cannot_resume(scratch_dir, filename):
    path, identity = make_checkpoint(scratch_dir)
    (path / filename).unlink()
    with pytest.raises(FileNotFoundError):
        contract.validate_resume(path, identity)


def test_same_size_loader_mutation_is_detected(scratch_dir):
    path, identity = make_checkpoint(scratch_dir)
    state = path / "data_0.pt"
    state.write_bytes(b"x" * state.stat().st_size)
    with pytest.raises(ValueError, match="inventory"):
        contract.validate_resume(path, identity)


@pytest.mark.parametrize("step", [3, 8])
def test_nonboundary_and_finished_runs_cannot_resume(scratch_dir, step):
    path, identity = make_checkpoint(scratch_dir, step=step)
    with pytest.raises(ValueError):
        contract.validate_resume(path, identity)


def test_resume_cannot_overwrite_later_checkpoint(scratch_dir):
    path, identity = make_checkpoint(scratch_dir)
    contract.save_run_identity(scratch_dir, identity)
    (scratch_dir / "global_step_8").mkdir()
    with pytest.raises(FileExistsError, match="NEW"):
        contract.validate_output(scratch_dir, identity, path)
    contract.validate_output(scratch_dir / "fresh", identity, path)


def test_identity_tracks_actual_files_and_effective_lr(monkeypatch, scratch_dir):
    _, args = arguments(monkeypatch, scratch_dir)
    train = scratch_dir / "train.parquet"
    train.write_bytes(b"first")
    prepared = {
        "input_sha256": "source",
        "train_rows": 32,
        "validation_rows": 0,
        "script_version": 5,
    }
    identity = contract.build_run_identity(
        args, prepared, train, None, {"files": {"model": "hash"}}, {}
    )
    assert identity["config"]["learning_rate"] == 1e-4
    train.write_bytes(b"other")
    changed = contract.build_run_identity(
        args, prepared, train, None, {"files": {"model": "hash"}}, {}
    )
    assert identity["data"] != changed["data"]


def test_runtime_rejects_wrong_checkout_without_importing_engine(
    monkeypatch, scratch_dir
):
    monkeypatch.setattr(
        contract.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(scratch_dir / "verl/__init__.py")),
    )
    replies = iter(["wrong-commit", ""])
    monkeypatch.setattr(
        contract.subprocess, "check_output", lambda *a, **k: next(replies)
    )
    with pytest.raises(RuntimeError, match="clean veRL"):
        contract.validate_runtime()


def test_launcher_fresh_resume_and_changed_data_preflight(monkeypatch, scratch_dir):
    module = load_launcher()
    base = scratch_dir / "base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    (base / "tokenizer_config.json").write_text(
        '{"chat_template":"fixed"}', encoding="utf-8"
    )
    (base / "model.safetensors").write_bytes(b"fixture-not-real-model-weights")
    train = scratch_dir / "train.parquet"
    train.write_bytes(b"original-prepared-content")
    output = scratch_dir / "output"
    prepared = {
        "input_sha256": "source",
        "train_rows": 32,
        "validation_rows": 0,
        "script_version": module.SCRIPT_VERSION,
    }
    monkeypatch.setattr(module, "resolve_model_snapshot", lambda *a, **kw: str(base))
    monkeypatch.setattr(module, "tokenizer_identity", lambda *a, **kw: {})
    monkeypatch.setattr(module, "prepare_dataset", lambda **kw: (train, None, prepared))
    monkeypatch.setattr(
        module, "validate_training_runtime", lambda: {"test_runtime": True}
    )
    launches = []
    monkeypatch.setattr(
        module.subprocess, "run", lambda command, **kw: launches.append(command)
    )
    flags = [
        "sft",
        "--output-dir",
        str(output),
        "--model",
        str(base),
        "--val-ratio",
        "0",
    ]
    monkeypatch.setattr(sys, "argv", flags)
    module.main()
    identity = json.loads((output / contract.RUN_MANIFEST).read_text(encoding="utf-8"))
    assert len(launches) == 1 and "model.lora_rank=64" in launches[0]
    checkpoint, _ = make_checkpoint(output, identity)
    monkeypatch.setattr(
        sys,
        "argv",
        flags + ["--resume-mode", "resume_path", "--resume-from-path", str(checkpoint)],
    )
    module.main()
    assert len(launches) == 2
    train.write_bytes(b"modified-prepared-content")
    with pytest.raises(ValueError, match="identity mismatch"):
        module.main()
    assert len(launches) == 2

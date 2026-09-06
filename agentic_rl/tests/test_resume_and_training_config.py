import json
from pathlib import Path

import pytest

from scripts.train_airline_grpo import build_command, prepare_run_directory
from tau2_agentic_rl.base_identity import capture_base_identity, save_base_identity
from tau2_agentic_rl.checkpoints import resolve_resume_path, restore_step_clock
from tau2_agentic_rl.config import load_yaml
from tau2_agentic_rl.training_config import effective_project_config


def project():
    return load_yaml(Path(__file__).parents[1] / "configs/rl/airline_grpo_v1.yaml")


def checkpoint(root):
    path = root / "outputs/runs/smoke/checkpoints/global_step_10"
    (path / "actor").mkdir(parents=True)
    (path / "data.pt").touch()
    for name in ("model", "optim", "extra_state"):
        (path / "actor" / f"{name}_world_size_1_rank_0.pt").touch()
    return path


def test_relative_resume_uses_project_not_verl_cwd(scratch_dir):
    path = checkpoint(scratch_dir)
    relative = path.relative_to(scratch_dir)
    assert resolve_resume_path(scratch_dir, relative) == path.resolve()
    command = build_command(
        project_root=scratch_dir,
        model_path="model",
        train_file=Path("train"),
        val_file=Path("val"),
        total_epochs=1,
        extra=[],
        resume_from_path=relative,
    )
    assert f"trainer.resume_from_path={path.resolve()}" in command


def test_incomplete_actor_checkpoint_fails_early(scratch_dir):
    path = checkpoint(scratch_dir)
    (path / "data.pt").unlink()
    with pytest.raises(FileNotFoundError, match="data.pt"):
        resolve_resume_path(scratch_dir, path)


def test_restore_uses_exact_checkpoint_counters_not_later_run_file(scratch_dir):
    path = checkpoint(scratch_dir)
    local = {"attempt_step": 15, "optimizer_step": 10, "consecutive_skips": 2}
    later = {"attempt_step": 30, "optimizer_step": 20, "consecutive_skips": 0}
    (path / "step_counters.json").write_text(json.dumps(local), encoding="utf-8")
    (path.parent.parent / "step_counters.json").write_text(
        json.dumps(later), encoding="utf-8"
    )
    clock = restore_step_clock(path, 10)
    assert (clock.attempt_step, clock.optimizer_step, clock.consecutive_skips) == (
        15,
        10,
        2,
    )
    (path / "step_counters.json").unlink()
    with pytest.raises(ValueError, match="do not match"):
        restore_step_clock(path, 10)


def test_yaml_hyperparameters_reach_real_hydra_command():
    config = project()
    config["optimizer"]["lr"] = 2e-6
    config["ppo"]["ppo_epochs"] = 3
    config["lora"]["rank"] = 16
    config["vllm"]["gpu_memory_utilization"] = 0.35
    command = build_command(
        project_root=Path("C:/test"),
        model_path="model",
        train_file=Path("train"),
        val_file=Path("dev"),
        total_epochs=1,
        extra=[],
        project_config=config,
    )
    assert "actor_rollout_ref.actor.optim.lr=2e-06" in command
    assert "actor_rollout_ref.actor.ppo_epochs=3" in command
    assert "actor_rollout_ref.model.lora_rank=16" in command
    assert "actor_rollout_ref.rollout.gpu_memory_utilization=0.35" in command


def test_extra_override_updates_runtime_snapshot_and_rejects_managed_identity():
    config = effective_project_config(
        project(), ["actor_rollout_ref.actor.optim.lr=0.000002"]
    )
    assert config["optimizer"]["lr"] == 2e-6
    with pytest.raises(ValueError, match="dedicated"):
        effective_project_config(project(), ["trainer.resume_from_path=elsewhere"])
    with pytest.raises(ValueError, match="pinned capped"):
        effective_project_config(project(), ["actor_rollout_ref.rollout.n=4"])


def resume_fixture(root):
    path = checkpoint(root)
    base = root / "base"
    base.mkdir()
    for name in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors",
    ):
        (base / name).write_text("{}", encoding="utf-8")
    save_base_identity(path.parent.parent, capture_base_identity(base))
    return path, base


@pytest.mark.parametrize("destination", ["same", "new_empty", "new_absent"])
def test_resume_destination_accepts_same_or_new_empty_run(scratch_dir, destination):
    path, base = resume_fixture(scratch_dir)
    target = path.parent.parent if destination == "same" else scratch_dir / "new"
    if destination == "new_empty":
        target.mkdir()
    runtime = prepare_run_directory(target, path, str(base), {"fixture": 1})
    assert load_yaml(runtime) == {"fixture": 1}
    assert (target / "base_model_identity.json").is_file()
    assert (path / "actor/model_world_size_1_rank_0.pt").is_file()


def test_cross_run_nonempty_destination_rejected_without_writes(scratch_dir):
    path, base = resume_fixture(scratch_dir)
    target = scratch_dir / "another"
    target.mkdir()
    (target / "keep.txt").write_text("original", encoding="utf-8")
    with pytest.raises(ValueError, match="nonempty"):
        prepare_run_directory(target, path, str(base), {})
    assert [p.name for p in target.iterdir()] == ["keep.txt"]
    assert (target / "keep.txt").read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize("failure", ["base", "config", "fresh_nonempty"])
def test_rejected_resume_or_fresh_run_does_not_modify_existing_files(
    scratch_dir, failure
):
    path, base = resume_fixture(scratch_dir)
    target = path.parent.parent
    if failure == "base":
        (base / "model.safetensors").write_text("changed", encoding="utf-8")
    if failure == "config":
        (target / "runtime_config.yaml").write_text("fixture: old\n", encoding="utf-8")
    before = {
        p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()
    }
    with pytest.raises((ValueError, FileExistsError)):
        prepare_run_directory(
            target,
            None if failure == "fresh_nonempty" else path,
            str(base),
            {"fixture": "new"},
        )
    after = {
        p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()
    }
    assert after == before


def test_prompt_override_has_one_runtime_and_hydra_source():
    config = effective_project_config(project(), ["data.max_prompt_length=7000"])
    assert config["rollout"]["initial_prompt_max_tokens"] == 7000
    with pytest.raises(ValueError, match="derived"):
        effective_project_config(
            project(), ["actor_rollout_ref.rollout.prompt_length=7000"]
        )

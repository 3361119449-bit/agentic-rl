import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_train30_12_steps as experiment


@pytest.mark.parametrize("incomplete_step6", [False, True])
@pytest.mark.parametrize("official_grpo", [False, True])
def test_two_milestones_and_no_automatic_evaluation_refill(
    monkeypatch, scratch_dir, incomplete_step6, official_grpo,
):
    name = "test-train30"
    run = scratch_dir / "outputs" / "runs" / name
    markers = []
    valid_samples = 79 if incomplete_step6 else 80
    milestones = (20, 40, 60) if official_grpo else (6, 12)
    monkeypatch.setenv("VERL_ROOT", str(scratch_dir / "verl"))
    monkeypatch.setattr(experiment, "current_profile", lambda *args: None)
    monkeypatch.setattr(experiment, "load_yaml", lambda *args: {})
    monkeypatch.setattr(experiment, "require_checkpoint_space", lambda *args, **kwargs: None)

    def checkpoint(step):
        return run / "checkpoints" / f"global_step_{step}"

    def verify(project, run_path, step):
        assert project == scratch_dir and run_path == run
        path = checkpoint(step)
        assert path.is_dir()
        return path

    def run_script(project, script, args, *, pause_at=0):
        assert project == scratch_dir
        if script == "train_airline_grpo.py":
            step = pause_at or milestones[-1]
            assert f"trainer.total_training_steps={milestones[-1]}" in args
            assert f"trainer.save_freq={milestones[0]}" in args
            assert "trainer.max_actor_ckpt_to_keep=null" in args
            assert ("--official-grpo-control" if official_grpo else "--reward-judge") in args
            if step != milestones[0]:
                prev = milestones[milestones.index(step) - 1]
                assert args[args.index("--resume-from-path") + 1] == str(checkpoint(prev))
            markers.append(step)
            checkpoint(step).mkdir(parents=True)
        elif script == "export_verl_lora.py":
            target = Path(args[args.index("--target-dir") + 1]) / "lora_adapter"
            target.mkdir()
            (target / "adapter_config.json").write_text("{}", encoding="utf-8")
            (target / "adapter_model.safetensors").write_bytes(b"weights")
        else:
            assert script == "evaluate_airline.py"
            assert "--no-reward-judge" in args
            assert args[args.index("--max-refill-rounds") + 1] == "0"
            assert args[args.index("--samples") + 1] == "4"
            tag = args[args.index("--tag") + 1]
            adapter = Path(args[args.index("--lora-adapter") + 1])
            markers.append(tag)
            target = scratch_dir / "outputs" / "evaluations" / tag
            target.mkdir(parents=True)
            (target / "evaluation_manifest.json").write_text(json.dumps({"identity": {
                "split": "official_test", "seed": experiment.SEED,
                "adapter_path": str(adapter.resolve()),
                "adapter_sha256": experiment.sha256_json(
                    experiment.fingerprint_directory(adapter)
                ),
                "truncation_policy": experiment.ZERO_TRUNCATION,
                "reward_judge_enabled": False, "user_sim_filter_enabled": False,
            }}), encoding="utf-8")

    monkeypatch.setattr(experiment, "verified_checkpoint", verify)
    monkeypatch.setattr(experiment, "run_script", run_script)
    monkeypatch.setattr(experiment, "summarize", lambda _: {
        "valid_samples": valid_samples, "infrastructure_failures": 0,
    })
    if incomplete_step6:
        with pytest.raises(ValueError, match="exactly 80"):
            experiment.run_experiment(scratch_dir, name, dry_run=False, resume=False,
                                      official_grpo=official_grpo)
        with pytest.raises(ValueError, match="exactly 80"):
            experiment.run_experiment(scratch_dir, name, dry_run=False, resume=True,
                                      official_grpo=official_grpo)
        assert markers == [milestones[0], f"{name}_step{milestones[0]}"]
    else:
        experiment.run_experiment(scratch_dir, name, dry_run=False, resume=False,
                                  official_grpo=official_grpo)
        experiment.run_experiment(scratch_dir, name, dry_run=False, resume=True,
                                  official_grpo=official_grpo)
        assert markers == [value for step in milestones for value in (step, f"{name}_step{step}")]
        preserved = (milestones[-1],) if official_grpo else milestones
        assert {path.name for path in checkpoint(milestones[-1]).parent.iterdir()} == {
            f"global_step_{step}" for step in preserved
        }


def test_fresh_experiment_checks_space_for_both_checkpoints_and_export(monkeypatch, scratch_dir):
    (scratch_dir / "model.safetensors").write_bytes(b"x" * 100)
    monkeypatch.setenv("MERGED_SFT_MODEL", str(scratch_dir))
    monkeypatch.setattr(experiment.shutil, "disk_usage", lambda _: SimpleNamespace(free=299))
    with pytest.raises(OSError, match="checkpoints and export"):
        experiment.require_checkpoint_space(scratch_dir / "run", 0, scratch_dir, "test")
    experiment.require_checkpoint_space(scratch_dir / "run", 6, scratch_dir, "test")

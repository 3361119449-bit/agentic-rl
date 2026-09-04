from pathlib import Path

from scripts.prepare_tau2_dataset import SMOKE_IDS
from scripts.train_airline_grpo import build_command


def test_smoke_is_disjoint_from_internal_dev() -> None:
    internal_dev = {"3", "7", "12", "23", "39", "43"}
    assert not set(SMOKE_IDS) & internal_dev


def test_rl_run_has_private_outputs_and_no_implicit_resume() -> None:
    root = Path("C:/project")
    run_root = root / "outputs/runs/smoke_lr5e-6_seed42"
    command = build_command(
        project_root=root,
        model_path="model",
        train_file=root / "train.parquet",
        val_file=root / "dev.parquet",
        total_epochs=1,
        extra=[],
        run_name="smoke_lr5e-6_seed42",
        run_root=run_root,
    )
    assert "trainer.resume_mode=disable" in command
    assert f"trainer.default_local_dir={run_root / 'checkpoints'}" in command
    assert "trainer.experiment_name=smoke_lr5e-6_seed42" in command

import json
from pathlib import Path

from scripts.build_annotations import DEFAULT_DEV_IDS
from scripts.prepare_tau2_dataset import SMOKE_IDS
from scripts.train_airline_grpo import build_command

ROOT = Path(__file__).resolve().parents[1]
APPROVED_RL_TASK_IDS = {
    "0",
    "1",
    "3",
    "5",
    "10",
    "11",
    "12",
    "14",
    "15",
    "17",
    "20",
    "21",
    "23",
    "28",
    "33",
    "34",
    "36",
    "38",
    "40",
    "41",
    "42",
    "43",
    "47",
    "49",
}
APPROVED_INTERNAL_DEV_IDS = {"4", "7", "9", "27", "39", "46"}


def split_data() -> dict[str, list[str]]:
    path = ROOT / "data/splits/airline_internal_dev.v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_rl_training_is_limited_to_approved_tasks() -> None:
    assert set(split_data()["rl_train"]) == APPROVED_RL_TASK_IDS


def test_annotation_builder_preserves_the_approved_internal_dev_split() -> None:
    assert set(DEFAULT_DEV_IDS) == APPROVED_INTERNAL_DEV_IDS
    assert set(split_data()["internal_dev"]) == APPROVED_INTERNAL_DEV_IDS


def test_smoke_is_a_subset_of_approved_rl_tasks() -> None:
    assert set(SMOKE_IDS) <= APPROVED_RL_TASK_IDS


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

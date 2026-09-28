from pathlib import Path

import pytest

from scripts.train_airline_grpo import build_command
from tau2_agentic_rl.config import load_yaml
from tau2_agentic_rl.training_config import effective_project_config

ROOT = Path(__file__).parents[1]


def config():
    cfg = load_yaml(ROOT / "configs/rl/airline_procredit_v2.yaml")
    cfg["credit"].update(version="procredit-turn-v3", process_credit="capped_local_max")
    cfg["reward"]["progress_version"] = "progress-v2"
    cfg["training_selection"] = {"mode": "verified_progress"}
    cfg["precision"] = {"model_dtype": "bfloat16", "param_dtype": "bfloat16",
                        "rollout_dtype": "bfloat16", "reduce_dtype": "float32",
                        "buffer_dtype": "float32"}
    return cfg


def test_v3_launch_explicitly_loads_and_computes_bf16():
    command = build_command(project_root=ROOT, model_path="/model",
                            train_file=Path("/train"), val_file=Path("/dev"),
                            total_epochs=1, extra=[], project_config=config())
    assert "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16" in command
    assert "actor_rollout_ref.actor.fsdp_config.dtype=bfloat16" in command
    assert "+actor_rollout_ref.actor.fsdp_config.mixed_precision={param_dtype:bfloat16,reduce_dtype:float32,buffer_dtype:float32}" in command
    assert "actor_rollout_ref.rollout.dtype=bfloat16" in command


@pytest.mark.parametrize("override", [
    "actor_rollout_ref.actor.fsdp_config.model_dtype=float16",
    "actor_rollout_ref.rollout.dtype=float16",
    "actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=float16",
    "actor_rollout_ref.actor.fsdp_config={model_dtype:float16}",
    "~actor_rollout_ref.rollout.dtype=null",
])
def test_conflicting_precision_overrides_are_rejected(override):
    with pytest.raises(ValueError, match="precision"):
        effective_project_config(config(), [override])


def test_v3_requires_training_filter_and_explicit_precision():
    for field in ("training_selection", "precision"):
        cfg = config()
        cfg.pop(field)
        with pytest.raises(ValueError):
            effective_project_config(cfg, [])


def test_v3_config_file_and_resume_bind_selection_and_precision(scratch_dir):
    from copy import deepcopy

    from tau2_agentic_rl.rl_resume import build_resume_identity

    cfg = load_yaml(ROOT / "configs/rl/airline_procredit_v3.yaml")
    effective_project_config(cfg, [])
    cfg["training_selection"]["identity"] = {"manifest_sha256": "source-A"}
    data = scratch_dir / "train.parquet"
    data.write_bytes(b"identity test only")
    command = build_command(project_root=ROOT, model_path="/model",
                            train_file=data, val_file=data, total_epochs=1,
                            extra=[], project_config=cfg)
    original = build_resume_identity(ROOT, cfg, command, stage="smoke")
    for field in ("precision", "training_selection"):
        changed = deepcopy(cfg)
        changed[field] = {"changed": True}
        assert build_resume_identity(ROOT, changed, command, stage="smoke") != original

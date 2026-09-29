from pathlib import Path

import pytest
from test_precision_v3 import config as v3_config

from scripts.train_airline_grpo import build_command
from tau2_agentic_rl.config import load_yaml, validate_procredit_config


def config():
    cfg = v3_config()
    cfg["reward"].update(mode="turn_local_v1", mandatory_policy_gate=False, task_safety_gate=False)
    cfg["credit"].update(version="procredit-turn-v4", unresolved_policy_credit="retry_frozen",
                         violation_sign="strictly_negative", process_credit="uncapped_local_additive")
    cfg["reward"].pop("process_penalty_cap", None)
    cfg["reward"].pop("over_turn_penalty_cap", None)
    cfg["reward"]["penalty_placement"] = "turn_only"
    cfg["slot_recovery"] = {"max_resamples": 2, "max_scoring_retries": 2, "on_exhaustion": "stop"}
    return cfg


def test_v4_launch_has_explicit_local_policy_and_slot_recovery():
    cfg = load_yaml(Path(__file__).parents[1] / "configs/rl/airline_procredit_v4.yaml")
    validate_procredit_config(cfg)
    command = build_command(project_root=Path(__file__).parents[1], model_path="/model",
                            train_file=Path("/train"), val_file=Path("/dev"),
                            total_epochs=1, extra=[], project_config=cfg)
    assert "+algorithm.procredit_enabled=true" in command
    assert "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16" in command


@pytest.mark.parametrize("section,key,value", [
    ("reward", "mandatory_policy_gate", True),
    ("reward", "task_safety_gate", True),
    ("credit", "turn_coefficient", 0),
    ("credit", "violation_sign", "unconstrained"),
    ("credit", "process_credit", "capped_local_max"),
    ("reward", "process_penalty_cap", .2),
    ("reward", "over_turn_penalty_cap", .08),
    ("reward", "penalty_placement", "outside_truncation"),
    ("slot_recovery", "max_resamples", -1),
    ("slot_recovery", "max_scoring_retries", True),
    ("slot_recovery", "on_exhaustion", "resample_group"),
])
def test_v4_rejects_silent_contract_regressions(section, key, value):
    cfg = config()
    cfg[section][key] = value
    with pytest.raises(ValueError):
        validate_procredit_config(cfg)

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.train_airline_grpo import build_command
from tau2_agentic_rl.config import load_yaml
from tau2_agentic_rl.training_config import effective_project_config

ROOT = Path(__file__).parents[1]


def project():
    return load_yaml(ROOT / "configs/rl/airline_procredit_v1.yaml")


def test_new_launcher_selects_turn_credit_without_changing_legacy():
    kwargs = dict(
        project_root=ROOT,
        model_path="/model",
        train_file=Path("/train"),
        val_file=Path("/dev"),
        total_epochs=1,
        extra=[],
    )
    command = build_command(**kwargs, project_config=project())
    assert "+algorithm.procredit_enabled=true" in command
    assert "algorithm.gamma=1.0" in command
    legacy = load_yaml(ROOT / "configs/rl/airline_grpo_v1.yaml")
    assert not any("procredit_enabled" in arg for arg in build_command(**kwargs, project_config=legacy))


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("judge", "enabled", False),
        ("credit", "gamma", 0.95),
        ("credit", "turn_centering", "token_mean"),
        ("credit", "trajectory_std", "sample"),
        ("credit", "process_penalty_in_turn_return", True),
        ("reward", "penalty_placement", "inside_truncation"),
        ("reward", "score_floor", -0.2),
        ("reward", "mandatory_policy_gate", False),
        ("dynamic_sampling", "criterion", "score_variance"),
    ],
)
def test_invalid_new_mode_config_fails_before_training(section, key, value):
    cfg = project()
    cfg[section][key] = value
    with pytest.raises(ValueError):
        effective_project_config(cfg, [])


def test_cli_cannot_turn_off_actual_credit_hook_or_change_discount():
    for arg in [
        "+algorithm.procredit_enabled=false",
        "algorithm.gamma=0.95",
        "algorithm={procredit_enabled:false}",
        "~algorithm.procredit_enabled=true",
        "algorithm.rollout_correction={bypass_mode:false}",
        "~algorithm.rollout_correction.bypass_mode=true",
    ]:
        with pytest.raises(ValueError):
            effective_project_config(project(), [arg])


def test_ablation_is_explicit_and_keeps_legacy_files_untouched():
    cfg = project()
    baseline = deepcopy(cfg)
    cfg["credit"]["turn_coefficient"] = 0
    result = effective_project_config(cfg, [])
    assert result["credit"]["turn_coefficient"] == 0
    assert baseline["credit"]["turn_coefficient"] == 1
    legacy = load_yaml(ROOT / "configs/rl/airline_grpo_v1.yaml")
    assert legacy["reward"]["normal_weights"]["db"] == 0.30


def test_v2_config_is_explicit_and_keeps_v1_reproducible():
    from tau2_agentic_rl.advantages import CreditConfig

    cfg = load_yaml(ROOT / "configs/rl/airline_procredit_v2.yaml")
    effective = effective_project_config(cfg, [])
    assert CreditConfig.from_project(effective).version == "procredit-turn-v2"
    assert effective["reward"]["mandatory_policy_gate"] is True
    assert effective["credit"]["violation_penalty"] == 1
    assert project()["credit"]["version"] == "procredit-turn-v1"
    cfg["credit"]["turn_centering"] = "valid_turn_mean"
    with pytest.raises(ValueError):
        effective_project_config(cfg, [])

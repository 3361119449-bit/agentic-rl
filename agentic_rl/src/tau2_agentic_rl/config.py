"""Configuration loading with explicit path and placeholder validation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

ENV_VALUE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}$")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and reject non-mapping roots."""
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} strings without silently keeping placeholders."""
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if not isinstance(value, str):
        return value
    match = ENV_VALUE.match(value)
    if match:
        name, default = match.groups()
        resolved = os.environ.get(name, default)
        if resolved is None or not resolved or resolved.startswith("FIX_EXACT_"):
            raise ValueError(f"Unresolved configuration value: {value}")
        return resolved
    expanded = os.path.expandvars(value)
    if "${" in expanded or expanded.startswith("FIX_EXACT_"):
        raise ValueError(f"Unresolved configuration value: {value}")
    return expanded


def load_runtime_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and resolve required environment variables."""
    config = load_yaml(path)
    validate_procredit_config(config)
    if config.get("judge", {}).get("enabled") is False:
        config["judge"] = {"enabled": False}
    return expand_env(config)


def validate_procredit_config(config: dict) -> None:
    """Fail before model/API work when the new algorithm would be ambiguous."""
    from tau2_agentic_rl.advantages import CreditConfig, is_procredit
    from tau2_agentic_rl.reward.score import build_reward_config

    if not is_procredit(config):
        return
    build_reward_config(config)
    CreditConfig.from_project(config)
    if config.get("judge", {}).get("enabled") is not True:
        raise ValueError("ProCredit requires Agent reward Judge; use legacy official-only otherwise")
    expected_credit = {
        "mode": "procredit_turn",
        "version": "procredit-turn-v1",
        "gamma": 1.0,
        "turn_centering": "valid_turn_mean",
        "trajectory_std": "population",
        "process_penalty_in_turn_return": False,
    }
    for key, expected in expected_credit.items():
        if config.get("credit", {}).get(key) != expected:
            raise ValueError(f"ProCredit requires credit.{key}={expected}")
    reward = config["reward"]
    if reward.get("score_floor") != 0 or reward.get("penalty_placement") != "outside_truncation":
        raise ValueError("ProCredit requires nonnegative scores and undiscounted penalties")
    if config.get("dynamic_sampling", {}).get("criterion") != "final_advantage_nonzero":
        raise ValueError("ProCredit requires final-advantage group filtering")

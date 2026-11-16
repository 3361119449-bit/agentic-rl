"""Configuration loading with explicit path and placeholder validation."""

from __future__ import annotations

import json
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


def current_profile(root: Path, purpose: str) -> Path:
    """Read the current profile for this entry point."""
    configs = root / "configs"
    profiles = json.loads((configs / "current.json").read_text(encoding="utf-8"))
    path = configs / profiles[purpose]
    if load_yaml(path).get("profile_status") != "current":
        raise ValueError(f"Current {purpose} profile is not marked current: {path}")
    return path


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
    from tau2_agentic_rl.budget import ContextBudget
    from tau2_agentic_rl.reward.score import build_reward_config

    tool_limit = config.get("rollout", {}).get("max_tool_calls_per_turn", 1)
    if type(tool_limit) is not int or tool_limit != 1:
        raise ValueError("rollout.max_tool_calls_per_turn must be exactly 1")

    if "max_context_length" in config.get("rollout", {}):
        ContextBudget.from_rollout(config["rollout"])

    judge = config.get("judge", {})
    mode = judge.get("mode", "per_turn")
    if mode not in {"per_turn", "trajectory_once_v1"}:
        raise ValueError("unknown judge.mode")
    if mode == "trajectory_once_v1" and (
        judge.get("calls_per_trajectory") != 1
        or config.get("credit", {}).get("final_turn_rule") != "simple-turn-v2"
    ):
        raise ValueError("judge trajectory_once_v1 requires one request and simple-turn-v2")
    if config.get("project", {}).get("annotation_version") == "v7-repaired-airline-train30":
        if mode != "trajectory_once_v1" or config.get("user_sim_filter", {}).get("enabled") is not False:
            raise ValueError("formal train30 requires judge trajectory_once_v1 and user_sim_filter disabled")

    reward = config.get("reward", {})
    if reward.get("task_score_mode") is not None and (
        not is_procredit(config)
        or reward.get("progress_version") != "progress-v3"
        or config.get("credit", {}).get("progress_credit") != "immediate_delta_v1"
    ):
        raise ValueError("task_score_mode requires progress-v3 and immediate ProCredit")

    if not is_procredit(config):
        if any(config.get("credit", {}).get(k) is not None for k in (
            "db_subgoal_credit", "post_completion_credit", "final_turn_rule", "progress_credit",
        )):
            raise ValueError("local goal credit requires ProCredit v4")
        return
    build_reward_config(config)
    credit = CreditConfig.from_project(config)
    if config.get("judge", {}).get("enabled") is not True:
        raise ValueError("ProCredit requires Agent reward Judge; use legacy official-only otherwise")
    expected_credit = {
        "mode": "procredit_turn",
        "version": credit.version,
        "gamma": 1.0,
        "turn_centering": (
            "none" if credit.progress_credit == "immediate_delta_v1" else
            "policy_attributed_turn_mean" if credit.version != "procredit-turn-v1"
            else "valid_turn_mean"
        ),
        "trajectory_std": "population",
        "process_penalty_in_turn_return": False,
    }
    for key, expected in expected_credit.items():
        if config.get("credit", {}).get(key) != expected:
            raise ValueError(f"ProCredit requires credit.{key}={expected}")
    local_only = credit.version == "procredit-turn-v4"
    if credit.version != "procredit-turn-v1" and config["credit"].get(
        "unresolved_policy_credit"
    ) != ("retry_frozen" if local_only else "no_positive_progress"):
        raise ValueError(
            "v4 unresolved policy evidence must retry frozen scoring"
            if local_only else "unresolved policy evidence must suppress positive progress credit"
        )
    if local_only:
        if config["credit"].get("violation_sign") != "strictly_negative":
            raise ValueError("v4 requires negative final advantages on every violation turn")
        recovery = config.get("slot_recovery", {})
        if recovery.get("on_exhaustion") != "salvage_group":
            raise ValueError("slot exhaustion must salvage scored group members")
        truncation_retries = recovery.get("max_truncation_resamples", 0)
        if type(truncation_retries) is not int or not 0 <= truncation_retries <= 2:
            raise ValueError("truncation resamples must be an integer within [0,2]")
        for key in ("max_resamples", "max_scoring_retries"):
            if type(recovery.get(key)) is not int or not 0 <= recovery[key] <= 10:
                raise ValueError(f"slot_recovery.{key} must be an integer within [0, 10]")
    if credit.version in {"procredit-turn-v3", "procredit-turn-v4"}:
        if config["reward"].get("progress_version") not in {"progress-v2", "progress-v3"}:
            raise ValueError("ProCredit v3/v4 requires versioned progress-v2 or progress-v3")
        expected_process = "uncapped_local_additive" if local_only else "capped_local_max"
        if config["credit"].get("process_credit") != expected_process:
            raise ValueError(f"{credit.version} requires {expected_process} process credit")
        selection_mode = "task_or_local_credit" if local_only else "verified_progress"
        if config.get("training_selection", {}).get("mode") != selection_mode:
            raise ValueError(f"{credit.version} requires {selection_mode} training selection")
        precision = dict(config.get("precision", {}))
        master_dtype = precision.pop("lora_master_dtype", None)
        if master_dtype not in (None, "float32"):
            raise ValueError("precision.lora_master_dtype must be float32 when set")
        if config.get("profile_status") == "current" and master_dtype != "float32":
            raise ValueError("current RL training requires FP32 LoRA master parameters")
        if precision != {
            "model_dtype": "bfloat16", "param_dtype": "bfloat16",
            "rollout_dtype": "bfloat16", "reduce_dtype": "float32", "buffer_dtype": "float32",
        }:
            raise ValueError("ProCredit v3/v4 requires explicit BF16 precision")
    if reward.get("score_floor") != 0 or reward.get("penalty_placement") != (
        "turn_only" if local_only else "outside_truncation"
    ):
        raise ValueError("ProCredit requires nonnegative scores and undiscounted penalties")
    if config.get("dynamic_sampling", {}).get("criterion") != "final_advantage_nonzero":
        raise ValueError("ProCredit requires final-advantage group filtering")

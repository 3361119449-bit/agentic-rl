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
    return expand_env(load_yaml(path))

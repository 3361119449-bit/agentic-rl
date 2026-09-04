"""Load versioned task annotations with strict task-id uniqueness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_task_mapping(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a list of task annotations keyed by ``task_id`` or ``id``."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"annotation root must be a list: {path}")
    result: dict[str, dict[str, Any]] = {}
    for row in payload:
        raw_id = row.get("task_id", row.get("id"))
        if raw_id is None:
            raise ValueError(f"annotation row has no task id: {row}")
        task_id = str(raw_id)
        if task_id in result:
            raise ValueError(f"duplicate annotation for task {task_id}")
        result[task_id] = row
    return result

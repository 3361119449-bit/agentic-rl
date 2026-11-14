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


def load_task_judge_guidance(
    root: Path, annotations: dict, task_ids: set[str]
) -> dict[str, str]:
    path = annotations.get("task_judge_guidance")
    if path is None:
        return {}
    rows = load_task_mapping(root / path)
    if rows.keys() != task_ids:
        raise ValueError("task Judge guidance must cover the configured task IDs")
    guidance = {task_id: row.get("guidance") for task_id, row in rows.items()}
    if any(not isinstance(text, str) or not text.strip() for text in guidance.values()):
        raise ValueError("each task Judge guidance entry must contain nonempty text")
    return guidance

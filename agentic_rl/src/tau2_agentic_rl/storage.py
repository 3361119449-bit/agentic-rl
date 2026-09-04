"""Atomic trajectory storage and offline iteration."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

from tau2_agentic_rl.schemas import TrajectoryRecord


class TrajectoryStore:
    """Store each trajectory as an atomic JSON file for safe rescoring."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, record: TrajectoryRecord) -> Path:
        """Atomically save one validated trajectory."""
        destination = self.root / f"{record.trajectory_id}.json"
        payload = record.model_dump_json(indent=2)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{record.trajectory_id}.",
            suffix=".tmp",
            dir=self.root,
            text=True,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return destination

    def records(self) -> Iterator[TrajectoryRecord]:
        """Yield all stored records in stable filename order."""
        for path in sorted(self.root.glob("*.json")):
            yield TrajectoryRecord.model_validate_json(path.read_text(encoding="utf-8"))


def append_metrics_jsonl(path: str | Path, metrics: dict) -> None:
    """Append one compact metrics record."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n")

"""Frozen evaluation identities, sample slots, and exact coverage checks."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from tau2_agentic_rl.pass_metrics import (
    validate_official_test_ids,
    validate_unit_score,
)
from tau2_agentic_rl.scoring_retry import scoring_pending
from tau2_agentic_rl.versions import sha256_file, sha256_json


@contextmanager
def evaluation_lock(root: Path):
    """Reject concurrent refill processes; a crashed process leaves an audit lock."""
    path = root / "evaluation.lock"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    try:
        yield
    finally:
        path.unlink()


def fingerprint_directory(path: Path) -> dict[str, str]:
    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
        and item.suffix in {".json", ".safetensors", ".bin", ".model", ".txt", ".jinja"}
    )
    if not files:
        raise FileNotFoundError(f"no model/adapter files: {path}")
    return {item.relative_to(path).as_posix(): sha256_file(item) for item in files}


def initialize_evaluation(
    root: Path, identity: dict[str, Any], *, resume: bool
) -> dict[str, Any]:
    _validate_sample_plan(identity)
    manifest = {"identity": identity, "manifest_id": sha256_json(identity)}
    path = root / "evaluation_manifest.json"
    if resume:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != manifest:
            raise ValueError("evaluation identity changed; use a new --tag")
        return saved
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"evaluation directory is not empty: {root}; use --resume or a new --tag"
        )
    root.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


def _validate_sample_plan(identity: dict[str, Any]) -> tuple[list[str], int]:
    raw_tasks = identity.get("task_ids")
    n = identity.get("samples_per_task")
    if not isinstance(raw_tasks, list) or type(n) is not int or n < 4:
        raise ValueError("evaluation needs unique tasks and at least four samples each")
    tasks = list(map(str, raw_tasks))
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("evaluation needs unique tasks and at least four samples each")
    if identity["split"] == "official_test":
        validate_official_test_ids(raw_tasks, identity.get("tau2_commit"))
        if n != 4 or identity["record_split"] != "test":
            raise ValueError("official test requires 20 tasks x 4 valid samples")
    return tasks, n


def evaluation_coverage(records_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    identity = manifest["identity"]
    if manifest["manifest_id"] != sha256_json(identity):
        raise ValueError("evaluation manifest hash mismatch")
    tasks, n = _validate_sample_plan(identity)
    expected = {(task, slot) for task in tasks for slot in range(n)}
    valid: dict[tuple[str, int], dict] = {}
    pending: dict[tuple[str, int], dict] = {}
    failures = 0
    trajectory_ids = set()
    for path in sorted(records_dir.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        metadata = row.get("metadata", {})
        if metadata.get("evaluation_manifest_id") != manifest["manifest_id"]:
            raise ValueError(f"foreign evaluation trajectory: {path.name}")
        slot = metadata.get("evaluation_sample_index")
        key = (str(row["task_id"]), slot)
        if (
            type(slot) is not int
            or key not in expected
            or row.get("split") != identity["record_split"]
        ):
            raise ValueError(f"unexpected task, split, or sample slot: {path.name}")
        if row["trajectory_id"] in trajectory_ids:
            raise ValueError("duplicate trajectory ID")
        trajectory_ids.add(row["trajectory_id"])
        # Validate any saved score before classifying the slot. Corrupt scores
        # must not become model failures or infrastructure retries silently.
        for field, score_key in (
            ("official_scores", "reward"),
            ("custom_reward", "strict_success"),
        ):
            score = row.get(field)
            if score is not None:
                if not isinstance(score, dict):
                    raise ValueError(f"{path.name}: {field} must be an object")
                validate_unit_score(
                    score.get(score_key), name=f"{path.name}: {field}.{score_key}"
                )
        if scoring_pending(row):
            if key in valid or key in pending:
                raise ValueError(f"duplicate interaction for valid slot: {key}")
            pending[key] = row
            continue
        if (
            row.get("custom_reward") is None
            or row.get("official_scores") is None
            or row.get("termination_reason")
            in {"infrastructure_error", "infrastructure_failure"}
        ):
            failures += 1
            continue
        if key in valid or key in pending:
            raise ValueError(f"duplicate valid sample for task/slot: {key}")
        valid[key] = row
    missing = sorted(
        expected - valid.keys() - pending.keys(), key=lambda key: (int(key[0]), key[1])
    )
    return {
        "complete": not missing and not pending,
        "expected_samples": len(expected),
        "valid_samples": len(valid),
        "infrastructure_failures": failures,
        "scoring_pending_records": list(pending.values()),
        "missing_slots": [
            {"task_id": task, "sample_index": slot} for task, slot in missing
        ],
        "records": list(valid.values()),
    }

"""Select verified progress tasks from the authorized training split only."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from tau2_agentic_rl.annotations import load_task_mapping
from tau2_agentic_rl.reward.progress import build_progress_trace
from tau2_agentic_rl.reward.required_actions import (
    load_action_dependencies,
    load_required_actions,
)
from tau2_agentic_rl.versions import sha256_file, sha256_json


def select_training_rows(
    *, rows: list[dict], tasks: dict, required: dict, dependencies: dict,
    transfers: dict, split: dict, evaluator=None,
) -> tuple[list[dict], dict]:
    """Evaluate the actual initial history; no rollout or model calls."""
    allowed = set(split["rl_train"])
    heldout = set(split["internal_dev"]) | set(split["official_test"])
    if allowed & heldout:
        raise ValueError("training selection split overlaps held-out data")
    selected, entries, seen = [], [], set()
    for row in rows:
        info = row["extra_info"]
        task_id = str(info["task_id"])
        if info.get("split") != "train" or task_id not in allowed or task_id in seen:
            raise ValueError("training selection requires unique approved train task IDs")
        if any(task_id not in source for source in (tasks, required, transfers)):
            raise ValueError(f"training selection missing task or annotation: {task_id}")
        seen.add(task_id)
        task = tasks[task_id]
        initial = task.get("initial_state") or {}
        messages = initial.get("message_history") or []
        trace = build_progress_trace(
            task=task, messages=messages, prefix_lengths=[len(messages)], events=[],
            required_actions=required[task_id], dependencies=dependencies.get(task_id, []),
            transfer_rule=transfers[task_id], initial_state_fingerprint=sha256_json(initial),
            evaluator=evaluator, version="progress-v2",
        )
        included = any(check["included"] for check in trace["checks"])
        entries.append({
            "task_id": task_id, "included": included,
            "reason": "verified_unmet_progress" if included else "no_unmet_progress",
            "checkset_fingerprint": trace["checkset_fingerprint"],
            "checks": trace["checks"],
        })
        if included:
            selected.append(row)
    if not selected:
        raise ValueError("training selection contains no verified progress tasks")
    return selected, {
        "version": "verified-progress-selection-v1",
        "source_rows_sha256": sha256_json(rows),
        "split_sha256": sha256_json(split),
        "included_task_ids": [item["task_id"] for item in entries if item["included"]],
        "excluded_task_ids": [item["task_id"] for item in entries if not item["included"]],
        "tasks": entries,
    }


def write_selection(source: Path, rows: list[dict], manifest: dict, output: Path) -> tuple[Path, dict]:
    """Write a separate immutable parquet; bind source and selection to resume."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    identity = {
        "source_sha256": sha256_file(source),
        "selected_rows_sha256": sha256_json(rows),
        "selection": manifest,
    }
    digest = sha256_json(identity)
    path = output / f"{digest}.parquet"
    sidecar = path.with_suffix(".json")
    if path.exists() or sidecar.exists():
        if not (path.is_file() and sidecar.is_file()):
            raise ValueError("incomplete training selection artifact")
        saved = json.loads(sidecar.read_text(encoding="utf-8"))
        if saved != {**identity, "parquet_sha256": sha256_file(path)}:
            raise ValueError("training selection artifact changed")
    else:
        output.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".selection-", suffix=".parquet", dir=output)
        try:
            with os.fdopen(fd, "wb") as handle:
                pq.write_table(pa.Table.from_pylist(rows), handle)
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        sidecar.write_text(json.dumps(
            {**identity, "parquet_sha256": sha256_file(path)}, indent=2, ensure_ascii=False,
        ) + "\n", encoding="utf-8")
    return path, {"manifest_sha256": sha256_file(sidecar), "selection_sha256": digest}


def prepare_training_selection(source: Path, project_root: Path, project: dict) -> tuple[Path, dict]:
    """Called only for the train parquet after the pinned Tau2 import path is set."""
    import pyarrow.parquet as pq
    from tau2.registry import registry

    task_list = registry.get_tasks_loader("airline")()
    tasks = {str(task.id): task.model_dump(mode="json") for task in task_list}
    annotations = project["annotations"]
    selected, manifest = select_training_rows(
        rows=pq.read_table(source).to_pylist(), tasks=tasks,
        required=load_required_actions(project_root / annotations["required_actions"]),
        dependencies=load_action_dependencies(project_root / annotations["action_dependencies"]),
        transfers=load_task_mapping(project_root / annotations["transfer_rules"]),
        split=json.loads((project_root / "data/splits/airline_internal_dev.v1.json").read_text(encoding="utf-8")),
    )
    manifest["tau2_commit"] = project["project"]["tau2_commit"]
    path, identity = write_selection(source, selected, manifest, source.parent / "procredit_v3")
    print("ProCredit train selection: " + json.dumps({
        "included": manifest["included_task_ids"],
        "excluded": manifest["excluded_task_ids"], "manifest": str(path.with_suffix(".json")),
    }))
    return path, identity

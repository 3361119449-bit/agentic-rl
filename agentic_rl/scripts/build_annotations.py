"""Build split-isolated, versioned annotations from reviewed source files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TRANSFER_TEXT = "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."
DEFAULT_DEV_IDS = ["3", "7", "12", "23", "39", "43"]


def _read(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _semantic_rows(tasks: list[dict[str, Any]], ids: set[str]) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        task_id = str(task["id"])
        if task_id not in ids:
            continue
        assertions = (task.get("evaluation_criteria") or {}).get("nl_assertions") or []
        rows.append(
            {
                "task_id": task_id,
                "semantic_checks": [
                    {
                        "criterion_id": f"{task_id}:nl:{index}",
                        "description": assertion,
                    }
                    for index, assertion in enumerate(assertions)
                ],
            }
        )
    return rows


def _policy_rows(ids: set[str]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": task_id,
            "deterministic_rules": [
                "confirmation_before_database_write",
                "one_tool_call_per_assistant_turn",
                "no_unannotated_database_mutation",
            ],
            "judge_checks": [],
        }
        for task_id in sorted(ids, key=int)
    ]


def _transfer_rows(
    actions_by_id: dict[str, list[dict[str, Any]]], ids: set[str]
) -> list[dict[str, Any]]:
    rows = []
    for task_id in sorted(ids, key=int):
        transfer_required = any(
            action.get("name") == "transfer_to_human_agents"
            for action in actions_by_id[task_id]
        )
        rows.append(
            {
                "task_id": task_id,
                "allowed": transfer_required,
                "required": transfer_required,
                "valid_conditions": (
                    [
                        {
                            "condition_id": f"{task_id}:policy_transfer",
                            "description": (
                                "The requested change cannot be completed by the airline "
                                "agent under the official policy and transfer is the proper resolution."
                            ),
                        }
                    ]
                    if transfer_required
                    else []
                ),
                "required_pre_transfer_action_groups": [],
                "required_communication_checks": (
                    [TRANSFER_TEXT] if transfer_required else []
                ),
                "semantic_checks": (
                    [
                        {
                            "criterion_id": f"{task_id}:transfer_valid",
                            "description": "The transfer is legitimate and not task avoidance.",
                        }
                    ]
                    if transfer_required
                    else []
                ),
            }
        )
    return rows


def build(source_dir: Path, output_dir: Path, split_output: Path) -> None:
    """Generate train/test-isolated annotation bundles and a fixed dev split."""
    tasks = _read(source_dir / "tasks.json")
    split = _read(source_dir / "split_tasks.json")
    compact = _read(source_dir / "airline_task_required_actions.json")
    actions_by_id = {str(row["id"]): row["actions"] for row in compact}
    task_ids = {str(task["id"]) for task in tasks}
    if set(actions_by_id) != task_ids:
        raise ValueError("required-action annotations do not cover exactly all tasks")
    if set(split["train"]) & set(split["test"]):
        raise ValueError("official train/test IDs overlap")
    if set(split["base"]) != task_ids:
        raise ValueError("official split does not cover exactly all task IDs")

    for split_name in ("train", "test"):
        ids = set(map(str, split[split_name]))
        compact_rows = [row for row in compact if str(row["id"]) in ids]
        _write(
            output_dir / f"airline_required_actions.{split_name}.v1.json", compact_rows
        )
        _write(
            output_dir / f"airline_semantic_checks.{split_name}.v1.json",
            _semantic_rows(tasks, ids),
        )
        _write(
            output_dir / f"airline_transfer_rules.{split_name}.v1.json",
            _transfer_rows(actions_by_id, ids),
        )
        _write(
            output_dir / f"airline_mandatory_policy_rules.{split_name}.v1.json",
            _policy_rows(ids),
        )

    train_ids = set(map(str, split["train"]))
    dev_ids = set(DEFAULT_DEV_IDS)
    if not dev_ids < train_ids:
        raise ValueError("internal-dev IDs must be a strict subset of official train")
    _write(
        split_output,
        {
            "schema_version": "1.0",
            "source": "official data/tau2/domains/airline/split_tasks.json",
            "rl_train": sorted(train_ids - dev_ids, key=int),
            "internal_dev": sorted(dev_ids, key=int),
            "official_test": sorted(map(str, split["test"]), key=int),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/annotations"),
    )
    parser.add_argument(
        "--split-output",
        type=Path,
        default=Path("data/splits/airline_internal_dev.v1.json"),
    )
    args = parser.parse_args()
    build(args.source_dir, args.output_dir, args.split_output)


if __name__ == "__main__":
    main()

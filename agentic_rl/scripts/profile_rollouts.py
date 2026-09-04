"""Stage -1 diagnostics for saved SFT baseline trajectories."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def profile_records(rows: list[dict[str, Any]], group_size: int = 8) -> dict[str, Any]:
    """Compute pre-training variance, quality, error and length diagnostics."""
    if not rows:
        raise ValueError("no trajectory records supplied")
    all_rows = rows
    rows = [
        row
        for row in all_rows
        if row.get("custom_reward") is not None
        and row.get("official_scores") is not None
    ]
    infrastructure_failures = len(all_rows) - len(rows)
    if not rows:
        raise ValueError("all supplied trajectories are infrastructure failures")
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)

    complete_groups: list[list[dict[str, Any]]] = []
    for task_rows in by_task.values():
        for start in range(0, len(task_rows), group_size):
            group = task_rows[start : start + group_size]
            if len(group) == group_size:
                complete_groups.append(group)

    rewards_by_group = [
        [float(row["custom_reward"]["train_reward"]) for row in group]
        for group in complete_groups
    ]
    all_same = [len(set(values)) == 1 for values in rewards_by_group]
    all_zero = [all(value == 0.0 for value in values) for values in rewards_by_group]
    all_one = [all(value == 1.0 for value in values) for values in rewards_by_group]

    train_rewards = [float(row["custom_reward"]["train_reward"]) for row in rows]
    turns = [float(row["assistant_turns"]) for row in rows]
    tokens = [float(row["trajectory_tokens"]) for row in rows]
    terminations = Counter(str(row["termination_reason"]) for row in rows)
    error_kinds = Counter(
        event["error_kind"]
        for row in rows
        for event in row.get("tool_events", [])
        if event.get("error_kind")
    )

    component_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for name, component in row["custom_reward"].get("components", {}).items():
            if component.get("applicable") and component.get("value") is not None:
                component_values[name].append(float(component["value"]))

    per_task = []
    for task_id, task_rows in sorted(by_task.items(), key=lambda item: int(item[0])):
        values = [float(row["custom_reward"]["train_reward"]) for row in task_rows]
        per_task.append(
            {
                "task_id": task_id,
                "sample_count": len(task_rows),
                "mean_train_reward": mean(values),
                "strict_success_rate": _rate(
                    [row["custom_reward"]["strict_success"] == 1.0 for row in task_rows]
                ),
                "official_success_rate": _rate(
                    [row["official_scores"]["reward"] == 1.0 for row in task_rows]
                ),
                "budget_exhausted_rate": _rate(
                    [
                        row["termination_reason"] == "budget_exhausted"
                        for row in task_rows
                    ]
                ),
                "transfer_rate": _rate(
                    [
                        row["custom_reward"]["branch"] == "human_transfer"
                        for row in task_rows
                    ]
                ),
                "mixed_reward_group": len(task_rows) == group_size
                and len(set(values)) > 1,
            }
        )

    return {
        "trajectory_count": len(rows),
        "attempted_trajectory_count": len(all_rows),
        "infrastructure_failure_count": infrastructure_failures,
        "task_count": len(by_task),
        "complete_group_count": len(complete_groups),
        "mean_train_reward": mean(train_rewards),
        "tau2_official_success_rate": _rate(
            [row["official_scores"]["reward"] == 1.0 for row in rows]
        ),
        "custom_strict_success_rate": _rate(
            [row["custom_reward"]["strict_success"] == 1.0 for row in rows]
        ),
        "all_same_group_rate": _rate(all_same),
        "all_zero_group_rate": _rate(all_zero),
        "all_one_group_rate": _rate(all_one),
        "mixed_group_rate": _rate([not value for value in all_same]),
        "component_means": {
            name: _mean(values) for name, values in sorted(component_values.items())
        },
        "termination_counts": dict(terminations),
        "tool_error_counts": dict(error_kinds),
        "turns": {
            "mean": mean(turns),
            "p50": _percentile(turns, 0.50),
            "p90": _percentile(turns, 0.90),
            "p99": _percentile(turns, 0.99),
        },
        "tokens": {
            "mean": mean(tokens),
            "p50": _percentile(tokens, 0.50),
            "p90": _percentile(tokens, 0.90),
            "p99": _percentile(tokens, 0.99),
        },
        "per_task": per_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records_dir", type=Path)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.records_dir.glob("*.json"))
    ]
    result = profile_records(rows, args.group_size)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

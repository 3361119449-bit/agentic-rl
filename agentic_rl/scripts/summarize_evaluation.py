"""Report only pass@1 and pass@4 from saved trajectory records."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _pass_at_k(successes: int, samples: int, k: int) -> float | None:
    if samples < k:
        return None
    if samples - successes < k:
        return 1.0
    return 1.0 - math.comb(samples - successes, k) / math.comb(samples, k)


def summarize(records_dir: Path) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(records_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        groups[str(record["task_id"])].append(record)
    if not groups:
        raise ValueError(f"no trajectory JSON files found in {records_dir}")

    per_task = []
    for task_id, rows in sorted(groups.items(), key=lambda item: int(item[0])):
        attempted = len(rows)
        rows = [
            row
            for row in rows
            if row.get("official_scores") is not None
            and row.get("custom_reward") is not None
        ]
        infrastructure_failures = attempted - len(rows)
        official = sum(
            (row.get("official_scores") or {}).get("reward") == 1.0 for row in rows
        )
        strict = sum(
            (row.get("custom_reward") or {}).get("strict_success") == 1.0
            for row in rows
        )
        per_task.append(
            {
                "task_id": task_id,
                "samples": len(rows),
                "attempted_samples": attempted,
                "infrastructure_failures": infrastructure_failures,
                "official_pass1": _pass_at_k(official, len(rows), 1),
                "official_pass4": _pass_at_k(official, len(rows), 4),
                "custom_strict_pass1": _pass_at_k(strict, len(rows), 1),
                "custom_strict_pass4": _pass_at_k(strict, len(rows), 4),
            }
        )

    keys = (
        "official_pass1",
        "official_pass4",
        "custom_strict_pass1",
        "custom_strict_pass4",
    )
    aggregate = {}
    for key in keys:
        values = [row[key] for row in per_task if row[key] is not None]
        aggregate[key] = sum(values) / len(values) if values else None
    return {
        "tasks": len(per_task),
        "infrastructure_failures": sum(
            row["infrastructure_failures"] for row in per_task
        ),
        "aggregate": aggregate,
        "per_task": per_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.records_dir)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

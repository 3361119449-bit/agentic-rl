"""Report Tau2 pass^1/pass^4 only for a complete, single-identity evaluation."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from tau2_agentic_rl.evaluation import evaluation_coverage


def pass_hat_k(samples: int, successes: int, k: int) -> float:
    """Tau2 consistency: all k trials succeed, not at least one (pass@k)."""
    if not 0 <= successes <= samples or k < 1:
        raise ValueError("require 0 <= successes <= samples and k >= 1")
    if samples < k:
        raise ValueError(f"pass^{k} needs at least {k} valid samples")
    return math.comb(successes, k) / math.comb(samples, k)


def summarize(records_dir: Path, *, allow_incomplete: bool = False) -> dict[str, Any]:
    manifest = json.loads(
        (records_dir.parent / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    coverage = evaluation_coverage(records_dir, manifest)
    records = coverage.pop("records")
    pending = coverage.pop("scoring_pending_records")
    coverage["scoring_pending_slots"] = [
        {
            "task_id": row["task_id"],
            "sample_index": row["metadata"]["evaluation_sample_index"],
        }
        for row in pending
    ]
    if not coverage["complete"]:
        if allow_incomplete:
            return {"status": "incomplete", **coverage}
        raise ValueError(
            "incomplete evaluation; no final metrics: "
            + json.dumps(
                {
                    "missing": coverage["missing_slots"],
                    "scoring_pending": coverage["scoring_pending_slots"],
                }
            )
        )
    groups: dict[str, list] = defaultdict(list)
    for row in records:
        groups[str(row["task_id"])].append(row)
    per_task = []
    for task, rows in sorted(groups.items(), key=lambda item: int(item[0])):
        # Match the pinned Tau2 is_successful tolerance.
        official = sum(
            1 - 1e-6 <= row["official_scores"]["reward"] <= 1 + 1e-6 for row in rows
        )
        strict = sum(row["custom_reward"]["strict_success"] == 1.0 for row in rows)
        per_task.append(
            {
                "task_id": task,
                "samples": len(rows),
                "official_pass1": pass_hat_k(len(rows), official, 1),
                "official_pass4": pass_hat_k(len(rows), official, 4),
                "custom_strict_pass1": pass_hat_k(len(rows), strict, 1),
                "custom_strict_pass4": pass_hat_k(len(rows), strict, 4),
            }
        )
    keys = (
        "official_pass1",
        "official_pass4",
        "custom_strict_pass1",
        "custom_strict_pass4",
    )
    return {
        "status": "complete",
        "metric_definition": "tau2_pass_hat_k",
        "metric_formula": "comb(successes, k) / comb(samples, k)",
        "manifest_id": manifest["manifest_id"],
        "tasks": len(per_task),
        **coverage,
        "aggregate": {
            key: sum(row[key] for row in per_task) / len(per_task) for key in keys
        },
        "per_task": per_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Only coverage, no pass metrics, while slots are missing",
    )
    args = parser.parse_args()
    result = summarize(args.records_dir, allow_incomplete=args.allow_incomplete)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

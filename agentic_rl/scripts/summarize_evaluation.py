"""Report Tau2 pass^1/pass^4 only for a complete, single-identity evaluation."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tau2_agentic_rl.evaluation import evaluation_coverage
from tau2_agentic_rl.pass_metrics import official_success
from tau2_agentic_rl.reward.truncation import TRUNCATED_TERMINATIONS, ZERO_TRUNCATION


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
    user_pending = coverage.pop("user_sim_pending_records")
    rejected = coverage.pop("user_sim_rejected_records")
    coverage["user_sim_pending_slots"] = [
        {
            "task_id": row["task_id"],
            "sample_index": row["metadata"]["evaluation_sample_index"],
        }
        for row in user_pending
    ]
    coverage["user_sim_rejection_reasons"] = dict(
        Counter(row["user_sim_result"]["violation_type"] for row in rejected)
    )
    with_judge = manifest["identity"].get("reward_judge_enabled", True)
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
                    "user_sim_pending": coverage["user_sim_pending_slots"],
                }
            )
        )
    groups: dict[str, list] = defaultdict(list)
    for row in records:
        groups[str(row["task_id"])].append(row)
    zero_policy = manifest["identity"].get("truncation_policy") == ZERO_TRUNCATION
    primary1, primary4 = (("evaluation_pass1", "evaluation_pass4") if zero_policy
                          else ("official_pass1", "official_pass4"))
    truncated_samples = sum(row.get("termination_reason") in TRUNCATED_TERMINATIONS
                            for row in records)
    per_task = []
    for task, rows in sorted(groups.items(), key=lambda item: int(item[0])):
        raw_official = sum(
            official_success(row["official_scores"]["reward"]) for row in rows
        )
        official = sum(official_success(row["official_scores"]["reward"])
                       and not (zero_policy and row.get("termination_reason") in TRUNCATED_TERMINATIONS)
                       for row in rows)
        strict = (
            sum(row.get("custom_reward") is not None and row["custom_reward"]["strict_success"] == 1.0
                and not (zero_policy and row.get("termination_reason") in TRUNCATED_TERMINATIONS)
                for row in rows)
            if with_judge
            else 0
        )
        per_task.append(
            {
                "task_id": task,
                "samples": len(rows),
                primary1: pass_hat_k(len(rows), official, 1),
                primary4: pass_hat_k(len(rows), official, 4),
                **({"raw_tau2_official_pass1": pass_hat_k(len(rows), raw_official, 1),
                    "raw_tau2_official_pass4": pass_hat_k(len(rows), raw_official, 4)}
                   if zero_policy else {}),
                **(
                    {
                        "custom_strict_pass1": pass_hat_k(len(rows), strict, 1),
                        "custom_strict_pass4": pass_hat_k(len(rows), strict, 4),
                    }
                    if with_judge
                    else {}
                ),
            }
        )
    keys = (
        primary1,
        primary4,
        "custom_strict_pass1",
        "custom_strict_pass4",
    )
    if not with_judge:
        keys = (primary1, primary4)
    if zero_policy:
        keys += ("raw_tau2_official_pass1", "raw_tau2_official_pass4")
    return {
        "status": "complete",
        "metric_definition": ("tau2_pass_hat_k_zero_truncation_v1" if zero_policy
                              else "tau2_pass_hat_k"),
        **({"truncation_policy": ZERO_TRUNCATION, "truncated_samples": truncated_samples,
            "primary_metrics": [primary1, primary4], "raw_official_metric_definition": "tau2_pass_hat_k"}
           if zero_policy else {}),
        "metric_formula": "comb(successes, k) / comb(samples, k)",
        "manifest_id": manifest["manifest_id"],
        "reward_judge_enabled": with_judge,
        "user_sim_filter_enabled": manifest["identity"].get(
            "user_sim_filter_enabled", False
        ),
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

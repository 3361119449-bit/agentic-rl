"""Screen ALL saved episodes into new accepted/rejected/pending audit directories.

This command never rerolls with an unknown current policy or touches input files.
Live DAPO/evaluation performs policy-bound replacements; historical data receives
an explicit replacement plan for use with its original policy/checkpoint.
"""

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from tau2_agentic_rl.config import expand_env, load_yaml
from tau2_agentic_rl.pass_metrics import TAU2_COMMIT
from tau2_agentic_rl.schemas import TrajectoryRecord
from tau2_agentic_rl.storage import TrajectoryStore
from tau2_agentic_rl.user_simulation import (
    FILTER_VERSION,
    build_user_sim_judge,
    check_user_simulation,
    make_user_sim_inputs,
    replacement_seed,
    validate_screen_inputs,
)
from tau2_agentic_rl.versions import sha256_json


def prepare_record(record, guidelines=None):
    if record.environment_transcript is None:
        raise ValueError(
            "full delivered transcript is required; actor context is not a substitute"
        )
    if record.user_sim_inputs is not None:
        return record
    if record.metadata.get("tau2_commit") != TAU2_COMMIT or not guidelines:
        raise ValueError(
            "legacy records need the pinned Tau2 version and --guidelines file"
        )
    # Only extract the scenario from the frozen task. Never send the task itself.
    task = (record.scoring_inputs or {}).get("judge", {}).get("task", {})
    if "user_scenario" not in task:
        raise ValueError("legacy record lacks a frozen user_scenario")
    record.user_sim_inputs = make_user_sim_inputs(
        guidelines,
        task["user_scenario"],
        record.environment_transcript,
    ).model_dump()
    record.metadata["user_sim_inputs_sha256"] = sha256_json(record.user_sim_inputs)
    return record


async def screen_records(records, judge, output):
    stores = {
        name: TrajectoryStore(output / name, attach_evaluation_identity=False)
        for name in ("accepted", "rejected", "pending")
    }
    reports = []
    pending = iter(records)

    class DeferredStore:
        def save(self, record):
            pass  # Final classification below writes exactly one complete audit.

    async def worker():
        for record in pending:
            try:
                valid = await check_user_simulation(record, judge, DeferredStore())
                state = "accepted" if valid else "rejected"
            except Exception as exc:
                state = "pending"
                record.metadata.update(
                    failure_phase="user_sim_judge", failure_message=str(exc)
                )
            stores[state].save(record)
            item = {
                "trajectory_id": record.trajectory_id,
                "task_id": record.task_id,
                "state": state,
                "user_sim_result": record.user_sim_result,
            }
            if state == "rejected":
                seed = record.environment_seed
                item["replacement"] = {
                    "policy_version": record.policy_version,
                    "new_environment_seed": replacement_seed(seed, 1)
                    if seed is not None
                    else None,
                    "fresh_user_cache_required": True,
                }
            reports.append(item)

    async with asyncio.TaskGroup() as group:
        for _ in range(min(4, len(records))):
            group.create_task(worker())
    return {
        "filter_version": FILTER_VERSION,
        "counts": dict(Counter(r["state"] for r in reports)),
        "records": sorted(reports, key=lambda r: r["trajectory_id"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("records_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--guidelines",
        type=Path,
        help="Exact historical simulator guidelines, required only for legacy records.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source, output = args.records_dir.resolve(), args.output_dir.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("source and output must be separate non-nested directories")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "use a new empty output directory; source audits are immutable"
        )
    guidelines = (
        args.guidelines.read_text(encoding="utf-8") if args.guidelines else None
    )
    paths = sorted(source.glob("*.json"))
    if not paths:
        raise ValueError("no saved trajectories found")
    records = [
        prepare_record(
            TrajectoryRecord.model_validate_json(p.read_text(encoding="utf-8")),
            guidelines,
        )
        for p in paths
    ]
    if len({r.trajectory_id for r in records}) != len(records):
        raise ValueError("duplicate trajectory IDs")
    for record in records:
        validate_screen_inputs(record.model_dump())
    if args.dry_run:
        print(
            json.dumps(
                {
                    "trajectories": len(records),
                    "api_calls": 0,
                    "input_allowlist_validated": True,
                }
            )
        )
        return
    project = load_yaml(args.config)
    # Resolve ONLY compliance settings, never require the Agent reward Judge.
    config = expand_env(project["user_sim_filter"])
    judge = build_user_sim_judge(
        {
            "user_sim_filter": config,
            "outputs": {"user_sim_judge_cache": str(output / "cache")},
        },
        output,
    )
    # The same per-request budget, including standalone offline screening.
    import os

    import yaml

    runtime = output / "runtime_config.yaml"
    runtime.write_text(
        yaml.safe_dump({"rollout": project["rollout"]}), encoding="utf-8"
    )
    os.environ["AGENTIC_RL_CONFIG"] = str(runtime)
    report = asyncio.run(screen_records(records, judge, output))
    (output / "screening_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["counts"]))


if __name__ == "__main__":
    main()

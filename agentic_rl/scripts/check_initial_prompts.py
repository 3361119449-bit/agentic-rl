"""Measure real Tau2 reset prompts, or summarize already measured rollout records.

Live mode uses the USER API, but never loads policy weights, generates an agent
answer, or calls the project's Judge. Tau2 may compute its built-in reward during
cleanup; this script does not report scores. No placeholder-prompt estimates.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

try:
    from scripts.prepare_tau2_dataset import SMOKE_IDS
    from scripts.train_airline_grpo import TAU2_COMMIT, _require_exact_checkout
except ModuleNotFoundError:
    from prepare_tau2_dataset import SMOKE_IDS
    from train_airline_grpo import TAU2_COMMIT, _require_exact_checkout

from tau2_agentic_rl.budget import ContextBudget
from tau2_agentic_rl.config import load_runtime_config
from tau2_agentic_rl.environment.tau2_gym import Tau2GymAdapter
from tau2_agentic_rl.initial_prompt import (
    encode_full_chat,
    initial_messages,
    inspect_initial_prompt,
    summarize_initial_prompts,
)

ROOT = Path(__file__).resolve().parents[1]


def select_tasks(split_file, split):
    data = json.loads(split_file.read_text(encoding="utf-8"))
    if split == "smoke":
        return list(SMOKE_IDS)
    if split == "official_train":
        return sorted(set(data["rl_train"] + data["internal_dev"]), key=int)
    return list(data[split])


def recorded_measurements(root, task_ids):
    rows = []
    for path in sorted(root.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if str(record["task_id"]) not in task_ids:
            continue
        row = record.get("metadata", {}).get("initial_prompt") or {
            "task_id": str(record["task_id"]),
            "status": "not_measured",
            "failure_phase": record.get("metadata", {}).get("failure_phase"),
        }
        rows.append({**row, "trajectory_id": record["trajectory_id"]})
    return rows


async def measure_live(args, task_ids, project, tokenizer):
    rollout, user = project["rollout"], project["user_simulator"]
    budget = ContextBudget(
        max_context_tokens=int(rollout["max_context_length"]),
        reserved_observation_tokens=int(rollout["reserved_observation_tokens"]),
        reserved_template_tokens=int(rollout["reserved_template_tokens"]),
        min_final_response_tokens=int(rollout["min_final_response_tokens"]),
        per_turn_max_new_tokens=int(rollout["per_turn_max_new_tokens"]),
    )
    rows = []
    for index, task_id in enumerate(task_ids):
        environment = Tau2GymAdapter(
            task_id=task_id,
            user_model=user["model"],
            user_temperature=float(user["temperature"]),
            user_llm_args=user.get("llm_args", {}),
            user_cache_dir=args.cache_dir,
            user_max_retries=int(user.get("max_retries", 2)),
            max_steps=int(rollout["max_hard_turns"]) * 3,
        )
        seed = index  # Matches prepare_tau2_dataset._row for this split.
        row = {"task_id": task_id, "environment_seed": seed}
        try:
            incoming = await environment.reset(seed=seed)
            ids = encode_full_chat(
                tokenizer,
                initial_messages(environment.policy, incoming),
                tools=environment.tool_schemas,
            )
            row.update(
                inspect_initial_prompt(
                    task_id, len(ids), int(rollout["initial_prompt_max_tokens"]), budget
                )
            )
        except Exception as exc:
            row.update(
                status="initialization_error",
                error_type=type(exc).__name__,
                error=str(exc),
            )
        finally:
            await environment.force_cleanup_stop()
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--records", type=Path, help="Offline: existing trajectory JSON directory"
    )
    mode.add_argument(
        "--live-user-api",
        action="store_true",
        help="Explicitly allow paid user-simulator reset calls",
    )
    parser.add_argument(
        "--split",
        choices=("smoke", "rl_train", "internal_dev", "official_train"),
        default="smoke",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=ROOT / "data/splits/airline_internal_dev.v1.json",
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/rl/airline_grpo_v1.yaml"
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MERGED_SFT_MODEL"),
        help="Local tokenizer/model snapshot; no weight loading",
    )
    parser.add_argument("--tau2-root", type=Path, default=os.environ.get("TAU2_ROOT"))
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / "outputs/prompt_preflight_user_cache"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(
            "choose a new --output; existing reports are not overwritten"
        )
    task_ids = select_tasks(args.split_file, args.split)
    if args.records is not None:
        if not args.records.is_dir():
            raise FileNotFoundError(args.records)
        rows = recorded_measurements(args.records, task_ids)
    else:
        if not args.model or args.tau2_root is None:
            parser.error("live mode requires a local --model and --tau2-root")
        _require_exact_checkout(args.tau2_root, TAU2_COMMIT, "Tau2")
        sys.path.insert(0, str(args.tau2_root.resolve() / "src"))
        os.environ["AGENTIC_RL_CONFIG"] = str(args.config.resolve())
        project = load_runtime_config(args.config)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        rows = asyncio.run(measure_live(args, task_ids, project, tokenizer))
    report = summarize_initial_prompts(rows, task_ids)
    report.update(
        split=args.split, source="live_reset" if args.live_user_api else "saved_records"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "rows"},
            ensure_ascii=False,
            indent=2,
        )
    )
    if not report["all_requested_samples_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run official Tau2 airline train/test splits with reproducible LLM settings."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_AGENT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_USER_MODEL = "deepseek/deepseek-v4-pro"
EXPECTED_TAU2_COMMIT = "a2c024725189473d2d7cea3a5cfdbcc67478e41f"
DEFAULT_DEEPSEEK_ARGS = {
    "temperature": 0.0,
    "extra_body": {"thinking": {"type": "disabled"}},
}


def json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return value


def safe_save_name(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or value.strip() in {"", "."}:
        raise argparse.ArgumentTypeError(
            "save name must be a non-empty relative path without '..'"
        )
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau2-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--save-name", type=safe_save_name, required=True)
    parser.add_argument("--num-trials", type=int, default=4)
    parser.add_argument("--agent-model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument("--user-model", default=DEFAULT_USER_MODEL)
    parser.add_argument(
        "--agent-llm-args",
        type=json_object,
        default=DEFAULT_DEEPSEEK_ARGS,
        metavar="JSON",
    )
    parser.add_argument(
        "--user-llm-args",
        type=json_object,
        default=DEFAULT_DEEPSEEK_ARGS,
        metavar="JSON",
    )
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--num-tasks", type=int)
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--verbose-logs", action="store_true")
    parser.add_argument(
        "--allow-unpinned-tau2",
        action="store_true",
        help="Allow a Tau2 commit other than the inspected pinned commit.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def uses_deepseek(model: str) -> bool:
    return model == "deepseek" or model.startswith("deepseek/")


def results_path(tau2_dir: Path, save_name: str) -> Path:
    configured = os.environ.get("TAU2_DATA_DIR")
    data_dir = Path(configured) if configured else tau2_dir / "data"
    if not data_dir.is_absolute():
        data_dir = tau2_dir / data_dir
    return data_dir / "simulations" / save_name / "results.json"


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        "uv",
        "run",
        "tau2",
        "run",
        "--domain",
        "airline",
        "--task-split-name",
        args.split,
        "--num-trials",
        str(args.num_trials),
        "--agent",
        "llm_agent",
        "--agent-llm",
        args.agent_model,
        "--agent-llm-args",
        json.dumps(args.agent_llm_args, separators=(",", ":")),
        "--user",
        "user_simulator",
        "--user-llm",
        args.user_model,
        "--user-llm-args",
        json.dumps(args.user_llm_args, separators=(",", ":")),
        "--max-concurrency",
        str(args.max_concurrency),
        "--max-steps",
        str(args.max_steps),
        "--seed",
        str(args.seed),
        "--save-to",
        args.save_name,
        "--enforce-communication-protocol",
    ]
    if args.num_tasks is not None:
        command.extend(["--num-tasks", str(args.num_tasks)])
    if args.task_ids:
        command.append("--task-ids")
        command.extend(args.task_ids)
    if args.auto_resume:
        command.append("--auto-resume")
    if args.verbose_logs:
        command.append("--verbose-logs")
    return command


def main() -> None:
    args = parse_args()
    tau2_dir = args.tau2_dir.resolve()
    if not (tau2_dir / "pyproject.toml").is_file() or not (
        tau2_dir / "src" / "tau2"
    ).is_dir():
        raise FileNotFoundError(f"not an official Tau2 source checkout: {tau2_dir}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tau2_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != EXPECTED_TAU2_COMMIT and not args.allow_unpinned_tau2:
        raise RuntimeError(
            f"Tau2 is at {commit}, expected {EXPECTED_TAU2_COMMIT}; "
            "checkout the pinned commit or explicitly use --allow-unpinned-tau2"
        )
    if args.num_trials < 1 or args.max_concurrency < 1 or args.max_steps < 1:
        raise ValueError("trials, concurrency, and max steps must be positive")
    if args.split == "test" and args.num_trials < 4:
        raise ValueError("the test split needs at least 4 trials to report pass^4")

    deepseek_needed = uses_deepseek(args.agent_model) or uses_deepseek(args.user_model)
    if deepseek_needed and not args.dry_run and not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is required and must not be put on the CLI")

    command = build_command(args)
    output = results_path(tau2_dir, args.save_name).resolve()
    print("Command:\n" + shlex.join(command))
    print(f"Tau2 commit: {commit}")
    print(f"Expected results: {output}")
    if args.dry_run:
        return
    subprocess.run(command, cwd=tau2_dir, check=True)
    if not output.is_file():
        raise RuntimeError(f"Tau2 completed but results were not found at {output}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

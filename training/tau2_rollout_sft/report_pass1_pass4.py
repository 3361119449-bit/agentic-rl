#!/usr/bin/env python3
"""Report only official Tau2 pass^1 and pass^4 from a results file."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def load_simulations(path: Path) -> list[dict[str, Any]]:
    path = path.resolve()
    metadata_path = path / "results.json" if path.is_dir() else path
    root = json.loads(metadata_path.read_text(encoding="utf-8"))
    simulations = root.get("simulations")
    if isinstance(simulations, list) and simulations:
        return simulations
    simulations_dir = metadata_path.parent / "simulations"
    if simulations_dir.is_dir():
        return [
            json.loads(simulation.read_text(encoding="utf-8"))
            for simulation in sorted(simulations_dir.glob("*.json"))
        ]
    return simulations if isinstance(simulations, list) else []


def successful(simulation: dict[str, Any]) -> bool:
    reward_info = simulation.get("reward_info")
    reward = reward_info.get("reward") if isinstance(reward_info, dict) else None
    return isinstance(reward, (int, float)) and math.isclose(
        float(reward), 1.0, abs_tol=1e-6
    )


def pass_hat_k(num_trials: int, successes: int, k: int) -> float:
    if num_trials < k:
        raise ValueError(f"{num_trials} usable trials are insufficient for pass^{k}")
    return math.comb(successes, k) / math.comb(num_trials, k)


def compute(simulations: list[dict[str, Any]]) -> dict[str, float]:
    by_task: defaultdict[str, list[bool]] = defaultdict(list)
    for simulation in simulations:
        if simulation.get("termination_reason") == "infrastructure_error":
            continue
        by_task[str(simulation.get("task_id"))].append(successful(simulation))
    if not by_task:
        raise ValueError("results contain no usable simulations")
    too_short = {task: len(values) for task, values in by_task.items() if len(values) < 4}
    if too_short:
        details = ", ".join(f"{task}:{count}" for task, count in sorted(too_short.items()))
        raise ValueError(f"cannot report pass^4; tasks with fewer than 4 trials: {details}")
    pass1 = []
    pass4 = []
    for values in by_task.values():
        n = len(values)
        c = sum(values)
        pass1.append(pass_hat_k(n, c, 1))
        pass4.append(pass_hat_k(n, c, 4))
    return {
        "pass^1": sum(pass1) / len(pass1),
        "pass^4": sum(pass4) / len(pass4),
    }


def write_text(path: Path, content: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    metrics = compute(load_simulations(args.results))
    payload = json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    print(payload, end="")
    if args.output_json:
        write_text(args.output_json, payload)
    if args.output_md:
        markdown = (
            "| metric | score |\n"
            "|---|---:|\n"
            f"| pass^1 | {metrics['pass^1']:.6f} |\n"
            f"| pass^4 | {metrics['pass^4']:.6f} |\n"
        )
        write_text(args.output_md, markdown)


if __name__ == "__main__":
    main()

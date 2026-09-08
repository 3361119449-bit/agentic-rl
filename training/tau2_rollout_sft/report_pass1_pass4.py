#!/usr/bin/env python3
"""Report only official Tau2 pass^1 and pass^4 from a results file."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

# Official airline test split at this repository's pinned Tau2 revision.
EXPECTED_TAU2_COMMIT = "a2c024725189473d2d7cea3a5cfdbcc67478e41f"
OFFICIAL_TEST_IDS = frozenset(
    {
        "2",
        "6",
        "8",
        "13",
        "16",
        "18",
        "19",
        "22",
        "24",
        "25",
        "26",
        "29",
        "30",
        "31",
        "32",
        "35",
        "37",
        "44",
        "45",
        "48",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def load_results(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = path.resolve()
    metadata_path = path / "results.json" if path.is_dir() else path
    root = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(root, dict):
        raise ValueError("Tau2 results root must be an object")
    simulations = root.get("simulations")
    if isinstance(simulations, list) and simulations:
        return root, simulations
    simulations_dir = metadata_path.parent / "simulations"
    if simulations_dir.is_dir():
        return root, [
            json.loads(simulation.read_text(encoding="utf-8"))
            for simulation in sorted(simulations_dir.glob("*.json"))
        ]
    return root, simulations if isinstance(simulations, list) else []


def validate_metadata(root: dict[str, Any]) -> int:
    """Do not label another domain, revision or task subset as official test."""
    info = root.get("info")
    if not isinstance(info, dict):
        raise ValueError("results need official Tau2 info metadata")
    environment = info.get("environment_info")
    if not isinstance(environment, dict) or environment.get("domain_name") != "airline":
        raise ValueError("only official airline test results may be reported")
    if info.get("git_commit") != EXPECTED_TAU2_COMMIT:
        raise ValueError("results must use the pinned Tau2 commit")
    tasks = root.get("tasks")
    if not isinstance(tasks, list) or any(not isinstance(task, dict) for task in tasks):
        raise ValueError("results must declare all official test tasks")
    task_ids = [task.get("id") for task in tasks]
    if (
        any(not isinstance(task, str) for task in task_ids)
        or len(task_ids) != len(OFFICIAL_TEST_IDS)
        or set(task_ids) != OFFICIAL_TEST_IDS
    ):
        raise ValueError("results tasks must be exactly the 20 official test IDs")
    trials = info.get("num_trials")
    if type(trials) is not int or trials < 4:
        raise ValueError("results must declare num_trials >= 4")
    return trials


def successful(simulation: dict[str, Any]) -> bool:
    reward_info = simulation.get("reward_info")
    reward = reward_info.get("reward") if isinstance(reward_info, dict) else None
    if (
        type(reward) not in (int, float)
        or not math.isfinite(reward)
        or not 0 <= reward <= 1
    ):
        raise ValueError("usable trials need a finite official reward in [0, 1]")
    return math.isclose(float(reward), 1.0, abs_tol=1e-6)


def pass_hat_k(num_trials: int, successes: int, k: int) -> float:
    if num_trials < k:
        raise ValueError(f"{num_trials} usable trials are insufficient for pass^{k}")
    return math.comb(successes, k) / math.comb(num_trials, k)


def compute(
    simulations: list[dict[str, Any]],
    *,
    num_trials: int = 4,
) -> dict[str, float]:
    """Require every declared test slot, allowing only infrastructure retries."""
    if type(num_trials) is not int or num_trials < 4:
        raise ValueError("num_trials must be an integer >= 4")
    by_task: dict[str, dict[int, bool]] = {task: {} for task in OFFICIAL_TEST_IDS}
    seen_ids: set[str] = set()
    for simulation in simulations:
        if not isinstance(simulation, dict):
            raise ValueError("each simulation must be an object")
        task = simulation.get("task_id")
        trial = simulation.get("trial")
        simulation_id = simulation.get("id")
        if not isinstance(task, str) or task not in by_task:
            raise ValueError(f"unexpected official test task ID: {task!r}")
        if type(trial) is not int or not 0 <= trial < num_trials:
            raise ValueError(f"invalid trial for task {task}: {trial!r}")
        if not isinstance(simulation_id, str) or not simulation_id.strip():
            raise ValueError("simulation ID must be a nonempty string")
        if simulation_id in seen_ids:
            raise ValueError(f"duplicate simulation ID: {simulation_id}")
        seen_ids.add(simulation_id)
        if simulation.get("termination_reason") == "infrastructure_error":
            continue
        if trial in by_task[task]:
            raise ValueError(f"duplicate usable task/trial: {task}/{trial}")
        by_task[task][trial] = successful(simulation)
    missing = {
        task: sorted(set(range(num_trials)) - values.keys())
        for task, values in sorted(by_task.items(), key=lambda item: int(item[0]))
        if len(values) != num_trials
    }
    if missing:
        raise ValueError(
            f"incomplete official test; no final metrics; missing slots: {missing}"
        )
    pass1 = []
    pass4 = []
    for values in by_task.values():
        n = len(values)
        c = sum(values.values())
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
    root, simulations = load_results(args.results)
    metrics = compute(simulations, num_trials=validate_metadata(root))
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

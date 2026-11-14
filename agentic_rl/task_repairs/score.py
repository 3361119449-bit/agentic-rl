"""Compare pinned Tau2 scores with the five repaired Airline tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.domains.airline.environment import get_environment, get_tasks
from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
from tau2.evaluator.evaluator_env import EnvironmentEvaluator

from tau2_agentic_rl.environment.booking_comparison import (
    booking_comparison_constructor,
    repaired_db_mode,
)

TASK_FILE = Path(__file__).with_name("tasks.json")


def load_repaired_tasks() -> dict[str, Task]:
    tasks = [Task.model_validate(row) for row in json.loads(TASK_FILE.read_bytes())]
    return {task.id: task for task in tasks}


def score_task(
    task: Task, simulation: SimulationRun, *, repaired: bool = False
) -> dict[str, float | None]:
    if simulation.task_id != task.id:
        raise ValueError("trajectory task_id must match the scoring task")
    if simulation.termination_reason not in {TerminationReason.USER_STOP, TerminationReason.AGENT_STOP}:
        return {"reward": 0.0, "db": None, "communicate": None}
    criteria = task.evaluation_criteria
    if criteria is None or set(criteria.reward_basis) != {"DB", "COMMUNICATE"}:
        raise ValueError("this scorer requires the DB + COMMUNICATE reward basis")
    messages = simulation.get_messages()
    constructor = booking_comparison_constructor(
        get_environment, repaired_db_mode(task.id) if repaired else None,
    )
    db = EnvironmentEvaluator.calculate_reward(
        constructor, task, messages, strict_replay=True
    ).db_check.db_reward
    communicate = CommunicateEvaluator.calculate_reward(task, messages).reward
    return {"reward": db * communicate, "db": db, "communicate": communicate}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simulation", type=Path, help="Native Tau2 SimulationRun JSON")
    args = parser.parse_args()
    simulation = SimulationRun.model_validate_json(args.simulation.read_bytes())
    repairs = load_repaired_tasks()
    if simulation.task_id not in repairs:
        parser.error("the repaired task IDs are 7, 14, 21, 23 and 39")
    original = next(task for task in get_tasks() if task.id == simulation.task_id)
    result = {
        "task_id": simulation.task_id,
        "original": score_task(original, simulation),
        "repaired": score_task(repairs[simulation.task_id], simulation, repaired=True),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Load the corrected definitions for the full Airline training set."""

from pathlib import Path

from tau2_agentic_rl.annotations import load_task_mapping
from tau2_agentic_rl.config import official_grpo_control


def load_training_task_repairs(root: Path, project: dict) -> dict:
    path = project.get("training_selection", {}).get("task_repairs")
    if path is None:
        return {}

    from tau2.data_model.tasks import Task

    if project["judge"]["enabled"] is not True and not official_grpo_control(project):
        raise ValueError("corrected Airline training requires the trajectory Judge")
    rows = load_task_mapping(root / path)
    if rows.keys() != {"7", "14", "21", "23", "39"}:
        raise ValueError("expected the five corrected Airline task definitions")
    return {
        task_id: Task.model_validate(row).model_dump(mode="json")
        for task_id, row in rows.items()
    }

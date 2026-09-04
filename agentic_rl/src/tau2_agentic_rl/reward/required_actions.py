"""Match independently selected completion actions against executed events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tau2_agentic_rl.schemas import (
    ComponentScore,
    RequiredActionResult,
    ToolEvent,
)

MUTATING_TOOLS = {
    "book_reservation",
    "cancel_reservation",
    "send_certificate",
    "update_reservation_baggages",
    "update_reservation_flights",
    "update_reservation_passengers",
}


def load_required_actions(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load the compact ``[{id, actions}]`` annotation file."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("required-actions root must be a list")

    result: dict[str, list[dict[str, Any]]] = {}
    for row in payload:
        if set(row) != {"id", "actions"}:
            raise ValueError("each required-actions row must contain only id/actions")
        task_id = str(row["id"])
        if task_id in result:
            raise ValueError(f"duplicate required-actions task id: {task_id}")
        if not isinstance(row["actions"], list):
            raise ValueError(f"actions for task {task_id} must be a list")
        result[task_id] = row["actions"]
    return result


def arguments_equal(expected: Any, actual: Any) -> bool:
    """Perform exact recursive equality for official action arguments."""
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(
            arguments_equal(expected[key], actual[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(expected) == len(actual) and all(
            arguments_equal(left, right)
            for left, right in zip(expected, actual, strict=True)
        )
    return expected == actual


def _event_satisfies(action: dict[str, Any], event: ToolEvent) -> bool:
    if event.name != action["name"] or not event.success:
        return False
    if not arguments_equal(action["arguments"], event.arguments):
        return False
    if event.name in MUTATING_TOOLS and event.db_effect is not True:
        return False
    return True


def evaluate_required_actions(
    required_actions: list[dict[str, Any]],
    events: list[ToolEvent],
) -> RequiredActionResult:
    """Greedily match each ordered required action to a later successful event."""
    required_ids = [str(action["action_id"]) for action in required_actions]
    if not required_actions:
        return RequiredActionResult(
            component=ComponentScore(applicable=False, value=None),
        )

    completed: list[str] = []
    matched_events: list[str] = []
    next_event_index = 0
    for action in required_actions:
        for index in range(next_event_index, len(events)):
            event = events[index]
            if _event_satisfies(action, event):
                completed.append(str(action["action_id"]))
                matched_events.append(event.event_id)
                next_event_index = index + 1
                break

    missing = [action_id for action_id in required_ids if action_id not in completed]
    return RequiredActionResult(
        component=ComponentScore(
            applicable=True,
            value=len(completed) / len(required_actions),
        ),
        required_action_ids=required_ids,
        completed_action_ids=completed,
        missing_action_ids=missing,
        matched_event_ids=matched_events,
    )

"""Match independently selected completion actions against executed events."""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
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

# Order is part of the itinerary, but it is not meaningful for passengers or
# payment allocations. Keep this list deliberately tool-specific.
UNORDERED_LIST_FIELDS: dict[str, set[tuple[str, ...]]] = {
    "book_reservation": {("passengers",), ("payment_methods",)},
    "update_reservation_passengers": {("passengers",)},
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


def _numeric_equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return False
    if not isinstance(expected, (int, float, str)) or not isinstance(
        actual, (int, float, str)
    ):
        return False
    try:
        return Decimal(str(expected)) == Decimal(str(actual))
    except InvalidOperation:
        return False


def arguments_equal(
    expected: Any,
    actual: Any,
    *,
    tool_name: str = "",
    path: tuple[str, ...] = (),
) -> bool:
    """Compare semantic tool arguments with narrowly scoped normalization."""
    if type(expected) is not type(actual):
        return _numeric_equal(expected, actual)
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(
            arguments_equal(
                expected[key],
                actual[key],
                tool_name=tool_name,
                path=(*path, str(key)),
            )
            for key in expected
        )
    if isinstance(expected, list):
        if path in UNORDERED_LIST_FIELDS.get(tool_name, set()):
            if len(expected) != len(actual):
                return False

            def match_unordered(position: int, used: set[int]) -> bool:
                if position == len(expected):
                    return True
                return any(
                    index not in used
                    and arguments_equal(
                        expected[position],
                        candidate,
                        tool_name=tool_name,
                        path=(*path, "*"),
                    )
                    and match_unordered(position + 1, {*used, index})
                    for index, candidate in enumerate(actual)
                )

            return match_unordered(0, set())
        return len(expected) == len(actual) and all(
            arguments_equal(
                left,
                right,
                tool_name=tool_name,
                path=(*path, str(index)),
            )
            for index, (left, right) in enumerate(zip(expected, actual, strict=True))
        )
    return expected == actual


def _event_satisfies(action: dict[str, Any], event: ToolEvent) -> bool:
    if event.name != action["name"] or not event.success:
        return False
    if not arguments_equal(action["arguments"], event.arguments, tool_name=event.name):
        return False
    if event.name in MUTATING_TOOLS and event.db_effect is not True:
        return False
    return True


def evaluate_required_actions(
    required_actions: list[dict[str, Any]],
    events: list[ToolEvent],
    dependencies: list[list[str]] | None = None,
) -> RequiredActionResult:
    """Find the largest one-to-one match while enforcing declared dependencies."""
    required_ids = [str(action["action_id"]) for action in required_actions]
    if not required_actions:
        return RequiredActionResult(
            component=ComponentScore(applicable=False, value=None),
        )

    action_ids = {str(action["action_id"]) for action in required_actions}
    dependency_pairs = [tuple(map(str, pair)) for pair in (dependencies or [])]
    for pair in dependency_pairs:
        if len(pair) != 2 or not set(pair) <= action_ids:
            raise ValueError(f"invalid required-action dependency: {pair}")

    candidates = {
        str(action["action_id"]): [
            index
            for index, event in enumerate(events)
            if _event_satisfies(action, event)
        ]
        for action in required_actions
    }
    ordered_actions = sorted(
        required_actions,
        key=lambda action: len(candidates[str(action["action_id"])]),
    )
    best: dict[str, int] = {}

    def dependency_ok(assignments: dict[str, int]) -> bool:
        return all(
            before not in assignments
            or after not in assignments
            or assignments[before] < assignments[after]
            for before, after in dependency_pairs
        )

    def dependency_complete(assignments: dict[str, int]) -> bool:
        return all(
            after not in assignments
            or (before in assignments and assignments[before] < assignments[after])
            for before, after in dependency_pairs
        )

    def search(position: int, assignments: dict[str, int], used: set[int]) -> None:
        nonlocal best
        if len(assignments) + len(ordered_actions) - position < len(best):
            return
        if position == len(ordered_actions):
            if dependency_complete(assignments) and len(assignments) > len(best):
                best = dict(assignments)
            return
        action_id = str(ordered_actions[position]["action_id"])
        for event_index in candidates[action_id]:
            if event_index in used:
                continue
            assignments[action_id] = event_index
            if dependency_ok(assignments):
                search(position + 1, assignments, {*used, event_index})
            assignments.pop(action_id)
        search(position + 1, assignments, used)

    search(0, {}, set())
    completed = [action_id for action_id in required_ids if action_id in best]
    matched_events = [events[best[action_id]].event_id for action_id in completed]

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


def load_action_dependencies(path: str | Path) -> dict[str, list[list[str]]]:
    """Load task-level ordering constraints without changing compact actions files."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    result: dict[str, list[list[str]]] = {}
    for row in payload:
        if set(row) != {"task_id", "dependencies"}:
            raise ValueError("dependency rows must contain task_id/dependencies")
        task_id = str(row["task_id"])
        if task_id in result:
            raise ValueError(f"duplicate dependency task id: {task_id}")
        result[task_id] = row["dependencies"]
    return result

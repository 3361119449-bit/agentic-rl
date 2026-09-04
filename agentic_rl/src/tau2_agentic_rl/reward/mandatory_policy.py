"""Deterministic hard policy checks for Airline trajectories."""

from __future__ import annotations

from collections import Counter
from typing import Any

from tau2_agentic_rl.reward.required_actions import MUTATING_TOOLS, arguments_equal
from tau2_agentic_rl.schemas import PolicyCheckResult, ToolEvent


def _matches_allowed_write(
    event: ToolEvent,
    required_actions: list[dict[str, Any]],
) -> bool:
    return any(
        action["name"] == event.name
        and arguments_equal(action["arguments"], event.arguments)
        for action in required_actions
    )


def evaluate_mandatory_policy(
    events: list[ToolEvent],
    required_actions: list[dict[str, Any]],
    judge_policy_checks: list[PolicyCheckResult] | None = None,
) -> list[PolicyCheckResult]:
    """Evaluate objective policy gates and append supplied semantic hard checks."""
    successful_writes = [
        event for event in events if event.success and event.name in MUTATING_TOOLS
    ]
    confirmation_failures = [
        event.event_id
        for event in successful_writes
        if event.confirmed_before is not True
    ]
    results = [
        PolicyCheckResult(
            rule_id="confirmation_before_database_write",
            applicable=bool(successful_writes),
            passed=not confirmation_failures,
            evidence_event_ids=confirmation_failures,
            reason=(
                "all database writes followed explicit user confirmation"
                if not confirmation_failures
                else "one or more database writes lacked prior confirmation"
            ),
        )
    ]

    calls_per_turn = Counter(event.turn_id for event in events if event.name)
    multi_call_turns = [turn for turn, count in calls_per_turn.items() if count > 1]
    results.append(
        PolicyCheckResult(
            rule_id="one_tool_call_per_assistant_turn",
            applicable=bool(calls_per_turn),
            passed=not multi_call_turns,
            reason=(
                "no assistant turn contains multiple tool calls"
                if not multi_call_turns
                else f"multiple calls occurred in turns {multi_call_turns}"
            ),
        )
    )

    unexpected_writes = [
        event.event_id
        for event in successful_writes
        if not _matches_allowed_write(event, required_actions)
    ]
    results.append(
        PolicyCheckResult(
            rule_id="no_unannotated_database_mutation",
            applicable=bool(successful_writes),
            passed=not unexpected_writes,
            evidence_event_ids=unexpected_writes,
            reason=(
                "all writes are allowed task-completion actions"
                if not unexpected_writes
                else "trajectory performed a write outside the allowed task actions"
            ),
        )
    )
    if judge_policy_checks:
        results.extend(judge_policy_checks)
    return results


def policy_gate_passed(checks: list[PolicyCheckResult]) -> bool:
    """Return true only when every applicable hard check passes."""
    return all(check.passed for check in checks if check.applicable)

"""Deterministic hard policy checks for Airline trajectories."""

from __future__ import annotations

from typing import Any

from tau2_agentic_rl.reward.required_actions import MUTATING_TOOLS, arguments_equal
from tau2_agentic_rl.schemas import PolicyCheckResult, ToolEvent


def matches_allowed_write(
    event: ToolEvent,
    required_actions: list[dict[str, Any]],
) -> bool:
    return any(
        action["name"] == event.name
        and arguments_equal(action["arguments"], event.arguments, tool_name=event.name)
        for action in required_actions
    )


def evaluate_mandatory_policy(
    events: list[ToolEvent],
    required_actions: list[dict[str, Any]],
    judge_policy_checks: list[PolicyCheckResult] | None = None,
) -> list[PolicyCheckResult]:
    """Evaluate objective policy gates and append supplied semantic hard checks."""
    write_events = [
        event
        for event in events
        if (event.success or event.db_effect is True) and event.name in MUTATING_TOOLS
    ]
    # The multitool SFT policy requires explicit confirmation before writes.
    # That conversational requirement is enforced by the frozen Judge rubric;
    # legacy confirmation-tracker fields remain audit-only here.
    results = []

    partial_writes = [event.event_id for event in write_events if not event.success]
    results.append(
        PolicyCheckResult(
            rule_id="failed_tool_with_database_mutation",
            applicable=bool(partial_writes),
            passed=not partial_writes,
            evidence_event_ids=partial_writes,
            reason="failed tools changed the database"
            if partial_writes
            else "no partial failed writes",
        )
    )

    if judge_policy_checks:
        results.extend(judge_policy_checks)
    return results


def evaluate_task_safety(
    events: list[ToolEvent], required_actions: list[dict[str, Any]]
) -> PolicyCheckResult:
    """Keep unannotated writes out of the official-policy gate namespace."""
    unexpected_writes = [
        event.event_id
        for event in events
        if (event.success or event.db_effect is True)
        and event.name in MUTATING_TOOLS
        and not matches_allowed_write(event, required_actions)
    ]
    return PolicyCheckResult(
        rule_id="no_unannotated_database_mutation",
        applicable=any(
            (event.success or event.db_effect is True) and event.name in MUTATING_TOOLS
            for event in events
        ),
        passed=not unexpected_writes,
        evidence_event_ids=unexpected_writes,
        reason=(
            "all writes match task-completion annotations"
            if not unexpected_writes
            else "trajectory performed a write outside task annotations"
        ),
    )


def policy_gate_passed(checks: list[PolicyCheckResult]) -> bool:
    """Return true only when every applicable hard check passes."""
    return all(check.passed for check in checks if check.applicable)

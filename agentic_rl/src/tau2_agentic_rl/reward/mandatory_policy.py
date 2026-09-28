"""Deterministic hard policy checks for Airline trajectories."""

from __future__ import annotations

from collections import Counter
from typing import Any

from tau2_agentic_rl.reward.required_actions import MUTATING_TOOLS
from tau2_agentic_rl.schemas import PolicyCheckResult, ToolEvent


def _write_scope(name: str, arguments: dict) -> tuple[str, str] | None:
    if name not in MUTATING_TOOLS:
        return None
    key = "user_id" if name in {"book_reservation", "send_certificate"} else "reservation_id"
    target = arguments.get(key)
    return (name, target) if isinstance(target, str) and target else None


def matches_allowed_write(
    event: ToolEvent,
    required_actions: list[dict[str, Any]],
) -> bool:
    """Check allowed write scope, independently of final completion arguments.

    Consent, eligibility and money rules remain Judge policy checks. Equality
    to the final reference state belongs to completion/DB, not this predicate.
    """
    scope = _write_scope(event.name, event.arguments)
    return scope is not None and any(
        _write_scope(action["name"], action["arguments"]) == scope for action in required_actions
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
    budgets = Counter(_write_scope(a["name"], a["arguments"]) for a in required_actions)
    creations = Counter()
    unexpected_writes = []
    for event in events:
        if not (event.success or event.db_effect is True) or event.name not in MUTATING_TOOLS:
            continue
        allowed = matches_allowed_write(event, required_actions)
        if event.name in {"book_reservation", "send_certificate"}:
            scope = _write_scope(event.name, event.arguments)
            creations[scope] += 1
            allowed = allowed and creations[scope] <= budgets[scope]
        if not allowed:
            unexpected_writes.append(event.event_id)
    return PolicyCheckResult(
        rule_id="no_unannotated_database_mutation",
        applicable=any(
            (event.success or event.db_effect is True) and event.name in MUTATING_TOOLS
            for event in events
        ),
        passed=not unexpected_writes,
        evidence_event_ids=unexpected_writes,
        reason=(
            "all writes stay within annotated tool/target scopes and creation budgets"
            if not unexpected_writes
            else "trajectory performed a write outside task annotations"
        ),
    )


def policy_gate_passed(checks: list[PolicyCheckResult]) -> bool:
    """Return true only when every applicable hard check passes."""
    return all(check.passed for check in checks if check.applicable)

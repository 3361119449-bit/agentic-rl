"""Human-transfer validation and component construction."""

from __future__ import annotations

from typing import Any

from tau2_agentic_rl.reward.required_actions import arguments_equal
from tau2_agentic_rl.schemas import ComponentScore, JudgeResult, ToolEvent

TRANSFER_MESSAGE = "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."


def successful_transfer_event(events: list[ToolEvent]) -> ToolEvent | None:
    """Find the first real successful transfer tool execution."""
    return next(
        (
            event
            for event in events
            if event.name == "transfer_to_human_agents" and event.success
        ),
        None,
    )


def _argument_pattern_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict) and set(expected) == {"eq"}:
        return arguments_equal(expected["eq"], actual)
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _argument_pattern_matches(value, actual[key])
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(expected) == len(actual)
            and all(
                _argument_pattern_matches(left, right)
                for left, right in zip(expected, actual, strict=True)
            )
        )
    return arguments_equal(expected, actual)


def _pre_transfer_score(
    events: list[ToolEvent],
    groups: list[dict[str, Any]],
    transfer: ToolEvent | None,
) -> float | None:
    if not groups:
        return None
    before = [
        event
        for event in events
        if transfer is None or event.sequence < transfer.sequence
    ]
    completed = 0
    for group in groups:
        matches = 0
        for event in before:
            for candidate in group.get("any_of", []):
                if event.name != candidate.get("tool"):
                    continue
                if group.get("require_tool_success", True) and not event.success:
                    continue
                if (
                    group.get("require_db_effect", False)
                    and event.db_effect is not True
                ):
                    continue
                if _argument_pattern_matches(
                    candidate.get("arguments", {}),
                    event.arguments,
                ):
                    matches += 1
                    break
        if matches >= int(group.get("min_count", 1)):
            completed += 1
    return completed / len(groups)


def _communication_score(
    messages: list[dict[str, Any]], checks: list[str | dict[str, Any]]
) -> float | None:
    if not checks:
        return None
    assistant_text = "\n".join(
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "assistant"
    ).lower()
    passed = 0
    for check in checks:
        expected = (
            check
            if isinstance(check, str)
            else check.get("contains", check.get("text", ""))
        )
        passed += bool(expected and str(expected).lower() in assistant_text)
    return passed / len(checks)


def transfer_components(
    events: list[ToolEvent],
    messages: list[dict[str, Any]],
    rule: dict[str, Any],
    judge: JudgeResult,
) -> tuple[bool, dict[str, ComponentScore]]:
    """Build transfer components and the transfer validity hard gate."""
    event = successful_transfer_event(events)
    allowed = bool(rule.get("allowed", False))
    judge_valid = (
        judge.transfer_check.valid if judge.transfer_check.applicable else True
    )
    valid = event is not None and allowed and judge_valid

    communication_checks = rule.get("required_communication_checks", [])
    communication_value = _communication_score(messages, communication_checks)
    semantic_applicable = bool(rule.get("semantic_checks"))
    semantic_checks = judge.transfer_semantic_checks
    semantic_value = (
        sum(item.passed for item in semantic_checks) / len(semantic_checks)
        if semantic_checks
        else None
    )
    pre_groups = rule.get("required_pre_transfer_action_groups", [])
    pre_value = _pre_transfer_score(events, pre_groups, event)
    components = {
        "transfer_call": ComponentScore(
            applicable=True, value=float(event is not None)
        ),
        "pre_transfer_actions": ComponentScore(
            applicable=bool(pre_groups),
            value=pre_value,
        ),
        "transfer_communication": ComponentScore(
            applicable=bool(communication_checks),
            value=communication_value,
        ),
        "transfer_semantic": ComponentScore(
            applicable=semantic_applicable,
            value=(semantic_value if semantic_applicable else None),
        ),
    }
    return valid, components

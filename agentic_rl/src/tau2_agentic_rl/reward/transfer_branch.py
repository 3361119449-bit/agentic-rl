"""Human-transfer validation and component construction."""

from __future__ import annotations

from typing import Any

from tau2_agentic_rl.failures import PolicyAttributionPending
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


def validate_transfer_verdict(judge: JudgeResult) -> None:
    """V2 needs fresh, explicit facts; legacy defaults are not decisions."""
    check = judge.transfer_check
    fields = {"applicable", "valid", "policy_required", "self_caused_only", "short_reason"}
    if (
        not fields <= check.model_fields_set
        or check.policy_required is None
        or check.self_caused_only is None
        or not check.short_reason.strip()
    ):
        raise PolicyAttributionPending(
            "transfer-policy-v2 requires an explicit transfer verdict; retry frozen Judge input"
        )


def transfer_required(rule: dict[str, Any], judge: JudgeResult, version: str) -> bool:
    """Required annotations and bound policy/user obligations override prohibition."""
    if version == "transfer-policy-v2":
        validate_transfer_verdict(judge)
        return bool(rule.get("required")) or judge.transfer_check.policy_required is True
    return bool(rule.get("required"))


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
    *,
    validation_version: str = "transfer-policy-v1",
) -> tuple[bool, dict[str, ComponentScore]]:
    """Build transfer components, independently of task completion."""
    event = successful_transfer_event(events)
    if validation_version == "transfer-policy-v2":
        required = transfer_required(rule, judge, validation_version)
        forbidden = bool(rule.get("forbidden")) and not required
        valid = (
            event is not None and not forbidden
            and judge.transfer_check.applicable and judge.transfer_check.valid
        )
    else:
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


def transfer_task_completion(components, valid, judge, validation_version):
    """Share the guarded transfer completion value with training readers."""
    from tau2_agentic_rl.reward.normal_branch import strict_success

    if validation_version == "transfer-policy-v2":
        validate_transfer_verdict(judge)
        if not valid or judge.transfer_check.self_caused_only:
            return 0.0
    return strict_success(components)

"""Localize policy violations without confusing supporting evidence with blame."""

from copy import deepcopy

from tau2_agentic_rl.schemas import JudgeResult, RewardResult, ToolEvent
from tau2_agentic_rl.versions import sha256_json


def annotate_policy_turns(
    messages: list[dict], prefix_lengths: list[int]
) -> list[dict]:
    if (
        not prefix_lengths
        or any(
            type(n) is not int or not 0 <= n <= len(messages) for n in prefix_lengths
        )
        or prefix_lengths != sorted(prefix_lengths)
        or prefix_lengths[-1] != len(messages)
    ):
        raise ValueError("invalid policy prefix boundaries")
    annotated = deepcopy(messages)
    for message in annotated:
        message.pop("assistant_turn_id", None)
    for turn, (start, end) in enumerate(
        zip(prefix_lengths[:-1], prefix_lengths[1:], strict=True), 1
    ):
        for message in annotated[start:end]:
            if message.get("role") == "assistant":
                message["assistant_turn_id"] = turn
    return annotated


def build_policy_credit(
    *,
    reward: RewardResult,
    judge: JudgeResult,
    events: list[ToolEvent],
    turns: int,
) -> dict:
    """Return zero-based violating actor turns and explicitly unresolved checks.

    Legacy evidence_turn_ids can cite user/tool context in an overlapping
    namespace. They are never treated as proof that a particular actor turn
    violated policy. Only deterministic event IDs and explicit blame IDs count.
    """
    evidence, unresolved, bad = [], set(), set()
    by_event = {event.event_id: event for event in events}

    def add(rule_id: str, turn_ids: list[int], source: str):
        if not turn_ids:
            unresolved.add(rule_id)
            return
        if any(type(t) is not int or not 1 <= t <= turns for t in turn_ids):
            raise ValueError("policy violation lies outside original assistant turns")
        indices = sorted({t - 1 for t in turn_ids})
        bad.update(indices)
        evidence.append({"rule_id": rule_id, "turns": indices, "source": source})

    invalid_transfer = (
        reward.branch == "human_transfer"
        and reward.details.get("transfer_valid") is False
    )
    transfer_turns = [
        event.turn_id
        for event in events
        if event.name == "transfer_to_human_agents" and event.success
    ]
    if invalid_transfer:
        add("invalid_transfer", transfer_turns, "executed_tool")
    failed_judge_ids = set()
    for check in judge.mandatory_policy_checks:
        if check.passed:
            continue
        failed_judge_ids.add(check.criterion_id)
        if check.violation_assistant_turn_ids:
            add(
                check.criterion_id,
                check.violation_assistant_turn_ids,
                "judge_explicit_blame",
            )
        elif invalid_transfer and check.criterion_id.endswith(
            ":policy:transfer_scope_and_message"
        ):
            add(check.criterion_id, transfer_turns, "executed_tool")
        else:
            unresolved.add(check.criterion_id)
    checks = list(reward.details.get("policy_checks", []))
    safety = reward.details.get("task_safety_check")
    if safety:
        checks.append(safety)
    for check in checks:
        rule_id = check["rule_id"]
        if check.get("passed", True) or rule_id in failed_judge_ids:
            continue
        ids = check.get("evidence_event_ids", [])
        if any(event_id not in by_event for event_id in ids):
            raise ValueError("policy violation refers to a missing tool event")
        add(rule_id, [by_event[event_id].turn_id for event_id in ids], "tool_event_id")
    valid = reward.policy_gate and reward.task_safety_gate
    if valid and (bad or unresolved):
        raise ValueError(
            "passing terminal gates disagree with policy violation evidence"
        )
    if not valid and not bad and not unresolved:
        unresolved.add("unattributed_gate_failure")
    inputs = {
        "judge": judge.model_dump(mode="json"),
        "events": [event.model_dump(mode="json") for event in events],
        "turns": turns,
        "valid": valid,
        "checks": checks,
        "invalid_transfer": invalid_transfer,
    }
    return {
        "version": "policy-credit-v1",
        "violating_turns": sorted(bad),
        "attribution_complete": not unresolved,
        "unresolved_checks": sorted(unresolved),
        "evidence": evidence,
        "inputs_fingerprint": sha256_json(inputs),
    }

"""Validate evidence against the exact data exposed to the trajectory Judge."""

from typing import Any

from tau2_agentic_rl.schemas import JudgeResult


def validate_evidence_turn_ids(result: JudgeResult, trajectory: dict[str, Any]) -> None:
    """Reject invented references without inferring IDs from array positions.

    Delivered messages use turn_idx; tool events use assistant-generation turn_id
    and can include rejected attempts absent from the delivered transcript.
    Existence in either supplied evidence view is required, not semantic support.
    """
    known_ids = {
        item[key]
        for section, key in (("messages", "turn_idx"), ("tool_events", "turn_id"))
        for item in trajectory.get(section, [])
        if type(item.get(key)) is int and item[key] >= 0
    }
    groups = (
        ("semantic_checks", result.semantic_checks),
        ("transfer_semantic_checks", result.transfer_semantic_checks),
        ("mandatory_policy_checks", result.mandatory_policy_checks),
        ("transfer_check", [result.transfer_check]),
    )
    for group, checks in groups:
        for check in checks:
            missing = sorted(set(check.evidence_turn_ids) - known_ids)
            if missing:
                criterion = getattr(check, "criterion_id", "transfer_check")
                raise ValueError(
                    f"judge {group}/{criterion} evidence_turn_ids reference absent "
                    f"turn IDs: {missing}; available IDs: {sorted(known_ids)}"
                )

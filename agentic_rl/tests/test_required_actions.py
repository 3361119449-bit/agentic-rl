from tau2_agentic_rl.reward.required_actions import evaluate_required_actions
from tau2_agentic_rl.schemas import ToolEvent


def event(
    sequence: int, reservation_id: str, *, success: bool = True, changed: bool = True
) -> ToolEvent:
    return ToolEvent(
        event_id=f"e{sequence}",
        sequence=sequence,
        turn_id=sequence,
        name="cancel_reservation",
        arguments={"reservation_id": reservation_id},
        success=success,
        db_effect=changed,
        confirmed_before=True,
    )


def test_required_actions_need_exact_arguments_order_success_and_effect() -> None:
    actions = [
        {
            "action_id": "a",
            "name": "cancel_reservation",
            "arguments": {"reservation_id": "A"},
        },
        {
            "action_id": "b",
            "name": "cancel_reservation",
            "arguments": {"reservation_id": "B"},
        },
    ]
    result = evaluate_required_actions(
        actions,
        [event(0, "B"), event(1, "A"), event(2, "B", changed=False), event(3, "B")],
    )
    assert result.component.value == 1.0
    assert result.matched_event_ids == ["e1", "e3"]


def test_empty_required_actions_are_inapplicable() -> None:
    result = evaluate_required_actions([], [])
    assert result.component.applicable is False
    assert result.component.value is None

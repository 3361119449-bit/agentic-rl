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


def test_independent_required_actions_can_run_in_either_order() -> None:
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
    assert result.matched_event_ids == ["e1", "e0"]


def test_declared_dependency_requires_event_order() -> None:
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
        dependencies=[["a", "b"]],
    )
    assert result.component.value == 1.0
    assert result.matched_event_ids == ["e1", "e3"]


def test_dependent_action_gets_no_credit_without_its_prerequisite() -> None:
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
        [event(0, "B")],
        dependencies=[["a", "b"]],
    )
    assert result.component.value == 0.0


def test_passenger_order_and_equivalent_numbers_are_normalized() -> None:
    action = {
        "action_id": "a",
        "name": "update_reservation_passengers",
        "arguments": {
            "reservation_id": "A",
            "passengers": [
                {"first_name": "Ada", "dob": "2000-01-01", "bags": 1},
                {"first_name": "Lin", "dob": "2001-01-01", "bags": 2},
            ],
        },
    }
    event_row = ToolEvent(
        event_id="e",
        sequence=0,
        turn_id=1,
        name=action["name"],
        arguments={
            "reservation_id": "A",
            "passengers": [
                {"first_name": "Lin", "dob": "2001-01-01", "bags": "2.0"},
                {"first_name": "Ada", "dob": "2000-01-01", "bags": 1.0},
            ],
        },
        success=True,
        db_effect=True,
    )
    assert evaluate_required_actions([action], [event_row]).component.value == 1.0


def test_empty_required_actions_are_inapplicable() -> None:
    result = evaluate_required_actions([], [])
    assert result.component.applicable is False
    assert result.component.value is None

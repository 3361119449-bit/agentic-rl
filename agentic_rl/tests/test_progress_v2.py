import pytest
from test_procredit_progress import build

from tau2_agentic_rl.reward.progress import validate_progress_trace
from tau2_agentic_rl.schemas import ToolEvent


def test_db_goal_is_not_counted_again_as_required_action():
    result = build(version="progress-v2", task={"id": "0"}, required_actions=[
        {"action_id": "write", "name": "cancel_reservation", "arguments": {"reservation_id": "R"}}
    ], events=[ToolEvent(event_id="w", sequence=0, turn_id=1, name="cancel_reservation",
                        arguments={"reservation_id": "R"}, success=True, db_effect=True)],
        evaluator=lambda task, messages: {"db": len(messages) == 2, "communicate": []})
    assert result["phi"] == [0, 1, 0]
    assert result["checks"][1]["check_id"] == "req:write"
    assert result["checks"][1]["included"] is False
    assert result["check_values"][-1] == [False, True]  # REQ remains auditable.
    validate_progress_trace(result, turns=2)
    result["checks"][1]["included"] = True
    with pytest.raises(ValueError):
        validate_progress_trace(result, turns=2)


def test_req_remains_progress_when_db_is_not_applicable():
    result = build(version="progress-v2", task={"id": "0"},
        required_actions=[{"action_id": "read", "name": "get_user_details", "arguments": {"user_id": "U"}}],
        events=[ToolEvent(event_id="r", sequence=0, turn_id=1, name="get_user_details",
                          arguments={"user_id": "U"}, success=True)],
        evaluator=lambda *args: {"db": None, "communicate": []})
    assert result["phi"] == [0, 1, 1]
    assert result["checks"][0]["included"]

from copy import deepcopy
from types import SimpleNamespace

import pytest

from tau2_agentic_rl.reward.mandatory_policy import evaluate_task_safety
from tau2_agentic_rl.reward.required_actions import (
    arguments_equal,
    evaluate_required_actions,
)
from tau2_agentic_rl.schemas import ToolEvent


def action_event(name, expected, actual=None, **metadata):
    action = {"action_id": "goal", "name": name, "arguments": expected}
    event = ToolEvent(event_id="e", sequence=0, turn_id=1, name=name,
                      arguments=deepcopy(expected if actual is None else actual),
                      success=True, db_effect=True, **metadata)
    return action, event


def test_flight_info_ignores_only_tool_discarded_metadata():
    args = {"reservation_id": "R", "cabin": "economy", "payment_id": "card",
            "flights": [{"flight_number": "A", "date": "2024-05-20"}]}
    actual = deepcopy(args)
    actual["flights"][0].update(origin="SFO", destination="JFK", price=123)
    action, event = action_event("update_reservation_flights", args, actual)
    assert evaluate_required_actions([action], [event]).component.value == 1
    event.arguments["flights"][0]["date"] = "2024-05-21"
    assert evaluate_required_actions([action], [event]).component.value == 0


def test_numeric_identifiers_are_not_coerced_like_amounts():
    assert not arguments_equal({"reservation_id": "001"}, {"reservation_id": 1},
                               tool_name="cancel_reservation")
    assert arguments_equal({"amount": 100}, {"amount": "100.0"}, tool_name="send_certificate")


def test_safety_scope_does_not_require_final_completion_payload():
    action, event = action_event("update_reservation_passengers",
        {"reservation_id": "R", "passengers": [{"first_name": "A", "last_name": "B", "dob": "2000-01-01"}]},
        {"reservation_id": "R", "passengers": [{"first_name": "A", "last_name": "C", "dob": "2000-01-01"}]})
    assert evaluate_required_actions([action], [event]).component.value == 0
    assert evaluate_task_safety([event], [action]).passed
    event.arguments["reservation_id"] = "UNAUTHORIZED"
    assert not evaluate_task_safety([event], [action]).passed


def test_safety_rejects_excess_append_only_writes():
    action, event = action_event("send_certificate", {"user_id": "U", "amount": 100})
    second = event.model_copy(update={"event_id": "e2", "sequence": 1, "turn_id": 2})
    check = evaluate_task_safety([event, second], [action])
    assert not check.passed
    assert check.evidence_event_ids == ["e2"]


@pytest.mark.parametrize("before,expected_match", [(1, True), (0, False), (None, False)])
def test_baggage_payment_equivalence_uses_actual_increment(before, expected_match):
    args = {"reservation_id": "R", "total_baggages": 3,
            "nonfree_baggages": 1, "payment_id": "cardA"}
    state = None if before is None else {"reservation_id": "R", "nonfree_baggages": before}
    action, event = action_event("update_reservation_baggages", args,
                                {**args, "payment_id": "cardB"}, state_before=state)
    assert bool(evaluate_required_actions([action], [event]).component.value) is expected_match


def test_baggage_snapshot_is_a_copy_and_not_an_extra_tool_call():
    from tau2_agentic_rl.environment.tau2_gym import snapshot_baggage_state

    reservation = SimpleNamespace(nonfree_baggages=1)
    backend = SimpleNamespace(tools=SimpleNamespace(db=SimpleNamespace(reservations={"R": reservation})))
    before = snapshot_baggage_state(backend, "update_reservation_baggages", {"reservation_id": "R"})
    reservation.nonfree_baggages = 2
    after = snapshot_baggage_state(backend, "update_reservation_baggages", {"reservation_id": "R"})
    assert before == {"reservation_id": "R", "nonfree_baggages": 1}
    assert after["nonfree_baggages"] == 2


def test_native_multitool_path_captures_state_before_each_call(monkeypatch):
    import sys
    import threading

    from tau2_agentic_rl.environment.tau2_gym import Tau2GymAdapter

    class Message(SimpleNamespace):
        def model_dump(self, **kwargs):
            return vars(self)

    monkeypatch.setitem(sys.modules, "tau2.data_model.message",
                        SimpleNamespace(AssistantMessage=Message, ToolCall=Message))
    reservation = SimpleNamespace(nonfree_baggages=1)

    def respond(call):
        reservation.nonfree_baggages = call.arguments["nonfree_baggages"]
        return Message(id=call.id, role="tool", content="ok", error=False)

    backend = SimpleNamespace(
        tools=SimpleNamespace(db=SimpleNamespace(reservations={"R": reservation})),
        get_db_hash=lambda: str(reservation.nonfree_baggages), get_response=respond,
    )
    agent = SimpleNamespace(observation=[], is_agent_turn=True)

    def act(action):
        agent.observation.append(action)
        agent.observation.extend(backend.get_response(call) for call in action.tool_calls)

    agent.set_action = act
    adapter = Tau2GymAdapter(task_id="0", user_model="fake")
    adapter.env = SimpleNamespace(
        _orchestrator=SimpleNamespace(environment=backend), _agent=agent,
        _simulation_done=threading.Event(), _lock=threading.Lock(),
        _get_reward=lambda: (0, {}), _get_info=lambda: {},
    )
    result = adapter._step_tools_sync([
        {"id": str(i), "name": "update_reservation_baggages",
         "arguments": {"reservation_id": "R", "nonfree_baggages": count}}
        for i, count in enumerate([2, 3])
    ])
    assert [item.state_before["nonfree_baggages"] for item in result.tool_results] == [1, 2]
    assert reservation.nonfree_baggages == 3
    assert backend.get_response is respond

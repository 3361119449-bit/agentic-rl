"""CPU-only native tool replay: no agent/user model and no rollout."""
import threading
from types import SimpleNamespace

import pytest

from tau2_agentic_rl.environment.tau2_gym import Tau2GymAdapter
from tau2_agentic_rl.reward import duplicate_queries
from tau2_agentic_rl.reward.duplicate_queries import (
    READ_ONLY_TOOLS,
    MissingQueryEvidence,
    capture_query_evidence,
    mark_duplicate_queries,
    query_state_projection,
)

pytest.importorskip("tau2.domains.airline.environment")
from tau2.domains.airline.environment import get_environment  # noqa: E402
from tau2.environment.toolkit import ToolType  # noqa: E402


class NativeDispatch:
    def __init__(self, backend):
        self.backend = backend
        self.observation = []
        self.is_agent_turn = True

    def set_action(self, action):
        self.is_agent_turn = False
        self.observation.append(action)
        for call in action.tool_calls:
            self.observation.append(self.backend.get_response(call))
        self.is_agent_turn = True


def adapter_for(backend):
    adapter = Tau2GymAdapter(task_id="0", user_model="unused")
    adapter.env = SimpleNamespace(
        _orchestrator=SimpleNamespace(environment=backend),
        _agent=NativeDispatch(backend), _simulation_done=threading.Event(),
        _lock=threading.Lock(), _get_reward=lambda: (0.0, {}),
        _get_info=lambda: {"simulation_run": {}, "tools": []},
    )
    return adapter


def test_pinned_read_allowlist_exactly_matches_native_tool_metadata():
    backend = get_environment()
    native_reads = {name for name in backend.tools.get_tools()
                    if backend.tools.tool_type(name) == ToolType.READ}
    assert native_reads | {"calculate"} == READ_ONLY_TOOLS
    assert backend.tools.tool_type("calculate") == ToolType.GENERIC
    assert not backend.tools.calculate.__mutates_state__


def test_adapter_captures_each_call_native_before_and_after_in_read_write_read_batch():
    backend = get_environment()
    user_id = next(iter(backend.tools.db.users))
    before = backend.get_db_hash()
    expected = capture_query_evidence(backend, "get_user_details", {"user_id": user_id})
    adapter = adapter_for(backend)
    step = adapter._step_tools_sync([
        {"id": "a", "name": "get_user_details", "arguments": {"user_id": user_id}},
        {"id": "b", "name": "send_certificate", "arguments": {"user_id": user_id, "amount": 1}},
        {"id": "c", "name": "get_user_details", "arguments": {"user_id": user_id}},
        {"id": "d", "name": "get_user_details", "arguments": {"user_id": user_id}},
    ])
    first, write, changed, duplicate = step.tool_results
    assert first.db_hash_before == first.db_hash_after == before
    assert write.success and write.db_hash_before == before
    assert write.db_hash_after != before
    assert changed.db_hash_before == write.db_hash_after == changed.db_hash_after
    assert first.query_state_before == expected["query_state_before"]
    assert first.query_state_before != changed.query_state_before
    assert changed.query_state_before == duplicate.query_state_before
    assert write.query_state_before is None
    events = [SimpleNamespace(**item.__dict__, event_id=item.call_id, sequence=i,
                              error_kind=None) for i, item in enumerate(step.tool_results)]
    assert [d["duplicate_query"] for d in mark_duplicate_queries(events, require_evidence=True)] == [
        False, False, False, True]
    assert first.result and user_id in first.result


def test_adapter_failed_native_read_cannot_establish_success():
    backend = get_environment()
    adapter = adapter_for(backend)
    step = adapter._step_tools_sync([
        {"id": "a", "name": "get_user_details", "arguments": {"user_id": "not-existing-user"}},
        {"id": "b", "name": "get_user_details", "arguments": {"user_id": "not-existing-user"}},
    ])
    events = [SimpleNamespace(**item.__dict__, event_id=item.call_id, sequence=i,
                              error_kind=None) for i, item in enumerate(step.tool_results)]
    assert all(not item.success and item.result for item in step.tool_results)
    assert all(not d["duplicate_query"] for d in mark_duplicate_queries(events, require_evidence=True))


def test_optimized_route_projection_equals_all_dates_reference_on_native_db(monkeypatch):
    backend = get_environment()

    def original_direct(flights, *, origin, destination, date, leave_after=None):
        rows = []
        for model in flights.values():
            flight = model.model_dump(mode="json")
            if flight["origin"] != origin:
                continue
            if destination is not None and flight["destination"] != destination:
                continue
            instance = flight["dates"].get(date)
            if instance is None or instance["status"] != "available":
                continue
            if leave_after is not None and flight["scheduled_departure_time_est"] < leave_after:
                continue
            rows.append({
                key: flight[key] for key in (
                    "flight_number", "origin", "destination",
                    "scheduled_departure_time_est", "scheduled_arrival_time_est",
                )
            } | {"date": date, "instance": instance})
        return rows

    # The reference keeps all-date serialization but uses the intended
    # complete-connection scope. Raw old one-stop dead-end projections are
    # intentionally superseded by this narrower relevant-state definition.
    # Include missing routes/dates and native overnight connection strings.
    cases = [
        ("SFO", "JFK", "2024-05-15"), ("JFK", "LAX", "2024-05-20"),
        ("LAX", "SEA", "2024-05-30"), ("SFO", "LAX", "2024-05-01"),
        ("NO_ROUTE", "JFK", "2024-05-15"), ("SFO", "JFK", "2099-01-01"),
    ]
    overnight = next(f for f in backend.tools.db.flights.values()
                     if "+1" in f.scheduled_arrival_time_est)
    date = next(d for d, item in overnight.dates.items() if item.status == "available")
    cases.append((overnight.origin, "JFK", date))
    for origin, destination, date in cases:
        args = {"origin": origin, "destination": destination, "date": date}
        for name in ("search_direct_flight", "search_onestop_flight"):
            actual_projection = query_state_projection(backend, name, args)
            actual_evidence = capture_query_evidence(backend, name, args)
            with monkeypatch.context() as patch:
                patch.setattr(duplicate_queries, "_direct_candidates", original_direct)
                assert actual_projection == query_state_projection(backend, name, args)
                assert actual_evidence == capture_query_evidence(backend, name, args)


def test_native_search_fingerprints_do_not_serialize_entire_flight_models(monkeypatch):
    backend = get_environment()
    flight = next(f for f in backend.tools.db.flights.values()
                  if any(item.status == "available" for item in f.dates.values()))
    date = next(d for d, item in flight.dates.items() if item.status == "available")
    args = {"origin": flight.origin, "destination": flight.destination, "date": date}
    expected = {name: capture_query_evidence(backend, name, args)
                for name in ("search_direct_flight", "search_onestop_flight")}

    def refuse_full_flight_dump(*args, **kwargs):
        raise AssertionError("query capture serialized every date in a native flight")

    monkeypatch.setattr(type(flight), "model_dump", refuse_full_flight_dump)
    for name, evidence in expected.items():
        assert capture_query_evidence(backend, name, args) == evidence


def test_adapter_detects_native_state_change_during_nominal_read(monkeypatch):
    backend = get_environment()
    user_id = next(iter(backend.tools.db.users))
    native_get_response = backend.get_response

    def response_with_intervening_write(call):
        response = native_get_response(call)
        backend.tools.send_certificate(user_id=user_id, amount=1)
        return response

    monkeypatch.setattr(backend, "get_response", response_with_intervening_write)
    step = adapter_for(backend)._step_tools_sync([
        {"id": "a", "name": "get_user_details", "arguments": {"user_id": user_id}},
    ])
    item = step.tool_results[0]
    assert item.success and item.db_changed
    assert item.db_hash_before != item.db_hash_after
    assert item.query_state_before != item.query_state_after
    event = SimpleNamespace(**item.__dict__, event_id=item.call_id, sequence=0, error_kind=None)
    with pytest.raises(MissingQueryEvidence, match="stable_read_state"):
        mark_duplicate_queries([event], require_evidence=True)


def test_native_onestop_deadend_edits_preserve_answer_and_fingerprint_until_viable():
    backend = get_environment()
    args = {"origin": "SFO", "destination": "JFK", "date": "2024-05-20"}
    dead = backend.tools.db.flights["HAT026"]
    original_answer = duplicate_queries._plain(backend.tools.search_onestop_flight(**args))
    assert not any(pair[0]["flight_number"] == dead.flight_number for pair in original_answer)
    original = capture_query_evidence(backend, "search_onestop_flight", args)
    instance = dead.dates[args["date"]]
    instance.available_seats["economy"] -= 1
    instance.prices["economy"] += 1
    assert duplicate_queries._plain(backend.tools.search_onestop_flight(**args)) == original_answer
    assert capture_query_evidence(backend, "search_onestop_flight", args) == original

    connection = dead.model_copy(deep=True)
    connection.flight_number = "NEW_VIABLE_CONNECTION"
    connection.origin = dead.destination
    connection.destination = args["destination"]
    connection.scheduled_departure_time_est = dead.scheduled_arrival_time_est
    date2 = (f"2024-05-{int(args['date'][-2:]) + 1}"
             if "+1" in dead.scheduled_arrival_time_est else args["date"])
    connection.dates = {date2: instance.model_copy(deep=True)}
    backend.tools.db.flights[connection.flight_number] = connection
    answer = duplicate_queries._plain(backend.tools.search_onestop_flight(**args))
    assert any(pair[0]["flight_number"] == dead.flight_number for pair in answer)
    assert answer != original_answer
    assert capture_query_evidence(backend, "search_onestop_flight", args) != original

from copy import deepcopy
from types import SimpleNamespace

import pytest

from tau2_agentic_rl.reward.duplicate_queries import (
    READ_ONLY_TOOLS,
    MissingQueryEvidence,
    canonical_arguments,
    capture_query_evidence,
    detect_duplicate_queries,
    mark_duplicate_queries,
)


def backend():
    return SimpleNamespace(tools=SimpleNamespace(db={
        "users": {"u": {"balance": 1}, "other": {"balance": 2}},
        "reservations": {"r": {"baggage": 1}, "other": {"baggage": 2}},
        "flights": {
            "F1": {"flight_number": "F1", "origin": "SFO", "destination": "JFK",
                   "scheduled_departure_time_est": "08:00:00",
                   "scheduled_arrival_time_est": "11:00:00",
                   "dates": {"2024-05-15": {"status": "available", "prices": {"economy": 10},
                                           "available_seats": {"economy": 3}},
                             "2024-05-16": {"status": "cancelled"}}},
            "F2": {"flight_number": "F2", "origin": "JFK", "destination": "LAX",
                   "scheduled_departure_time_est": "12:00:00",
                   "scheduled_arrival_time_est": "15:00:00",
                   "dates": {"2024-05-15": {"status": "available", "prices": {"economy": 20},
                                           "available_seats": {"economy": 4}}}},
        },
    }))


def event(i, b=None, *, name="get_user_details", args=None, success=True):
    args = {"user_id": "u"} if args is None else args
    proof = capture_query_evidence(b, name, args) if b is not None else {}
    return SimpleNamespace(
        event_id=f"e{i}", sequence=i, turn_id=i, name=name, arguments=args,
        success=success, result="native result", error_kind=None, execution_status=None,
        db_hash_before=f"global{i}", db_hash_after=f"global{i}", **proof,
        duplicate_query=False, duplicate_of_event_id=None,
    )


def test_successful_duplicates_two_and_three_reference_first_with_same_relevant_state():
    b = backend()
    events = [event(0, b), event(1, b), event(2, b)]
    audits = mark_duplicate_queries(events, require_evidence=True)
    assert [e.duplicate_query for e in events] == [False, True, True]
    assert [e.duplicate_of_event_id for e in events] == [None, "e0", "e0"]
    assert audits[2]["status"] == "duplicate"
    assert events[1].result == "native result"


def test_unrelated_user_and_reservation_changes_do_not_allow_refresh():
    b = backend()
    first = event(0, b)
    b.tools.db["users"]["other"]["balance"] += 10
    b.tools.db["reservations"]["r"]["baggage"] += 1
    second = event(1, b)
    assert detect_duplicate_queries([first, second], require_evidence=True)[1]["duplicate_query"]


def test_related_user_change_and_reverting_state():
    b = backend()
    first = event(0, b)
    b.tools.db["users"]["u"]["balance"] += 10
    second = event(1, b)
    b.tools.db["users"]["u"]["balance"] -= 10
    third = event(2, b)
    assert [x["duplicate_query"] for x in detect_duplicate_queries([first, second, third])] == [
        False, False, True,
    ]


def test_failed_service_reads_do_not_establish_successful_prior():
    b = backend()
    events = [event(0, b, success=False), event(1, b), event(2, b, success=False), event(3, b)]
    assert [x["duplicate_query"] for x in detect_duplicate_queries(events)] == [False, False, False, True]


def test_argument_normalization_keeps_native_string_and_list_semantics():
    assert canonical_arguments({"b": 2, "a": {"y": 1, "x": 0}}) == canonical_arguments(
        {"a": {"x": 0, "y": 1}, "b": 2})
    assert canonical_arguments({"a": ["a", "b"]}) != canonical_arguments({"a": ["b", "a"]})
    assert canonical_arguments({"user_id": "u"}) != canonical_arguments({"user_id": " U "})
    assert canonical_arguments({"x": True}) != canonical_arguments({"x": 1})


@pytest.mark.parametrize(("name", "args"), [
    ("get_user_details", {"user_id": "u"}),
    ("get_reservation_details", {"reservation_id": "r"}),
    ("get_flight_status", {"flight_number": "F1", "date": "2024-05-15"}),
    ("search_direct_flight", {"origin": "SFO", "destination": "JFK", "date": "2024-05-15"}),
    ("search_onestop_flight", {"origin": "SFO", "destination": "LAX", "date": "2024-05-15"}),
    ("list_all_airports", {}),
    ("calculate", {"expression": "2 + 2"}),
])
def test_all_native_read_tools_have_proof_and_are_detected(name, args):
    b = backend()
    assert name in READ_ONLY_TOOLS
    assert detect_duplicate_queries([event(0, b, name=name, args=args),
                                     event(1, b, name=name, args=args)],
                                    require_evidence=True)[1]["duplicate_query"]


@pytest.mark.parametrize("name", ["book_reservation", "cancel_reservation", "send_certificate",
                                  "update_reservation_baggages", "update_reservation_flights",
                                  "update_reservation_passengers", "transfer_to_human_agents", "done"])
def test_all_writes_and_control_excluded_even_without_proof(name):
    assert not detect_duplicate_queries([event(0, name=name), event(1, name=name)],
                                       require_evidence=True)[1]["duplicate_query"]


def test_reservation_scope_related_and_unrelated_changes():
    b = backend()
    args = {"reservation_id": "r"}
    first = event(0, b, name="get_reservation_details", args=args)
    b.tools.db["reservations"]["other"]["baggage"] = 20
    same = event(1, b, name=first.name, args=args)
    b.tools.db["reservations"]["r"]["baggage"] = 20
    changed = event(2, b, name=first.name, args=args)
    assert [x["duplicate_query"] for x in detect_duplicate_queries([first, same, changed])] == [
        False, True, False]


def test_search_relevant_date_route_changes_and_unrelated_dates():
    b = backend()
    args = {"origin": "SFO", "destination": "JFK", "date": "2024-05-15"}
    first = event(0, b, name="search_direct_flight", args=args)
    b.tools.db["flights"]["F1"]["dates"]["2024-05-16"]["status"] = "flying"
    b.tools.db["flights"]["F2"]["dates"]["2024-05-15"]["prices"]["economy"] = 100
    same = event(1, b, name=first.name, args=args)
    b.tools.db["flights"]["F1"]["dates"]["2024-05-15"]["prices"]["economy"] = 100
    changed = event(2, b, name=first.name, args=args)
    assert [x["duplicate_query"] for x in detect_duplicate_queries([first, same, changed])] == [
        False, True, False]


def test_onestop_relevant_connection_change_and_new_route_membership():
    b = backend()
    args = {"origin": "SFO", "destination": "LAX", "date": "2024-05-15"}
    first = event(0, b, name="search_onestop_flight", args=args)
    b.tools.db["flights"]["F2"]["dates"]["2024-05-15"]["prices"]["economy"] = 100
    changed = event(1, b, name=first.name, args=args)
    assert not detect_duplicate_queries([first, changed])[1]["duplicate_query"]
    b.tools.db["flights"]["F3"] = deepcopy(b.tools.db["flights"]["F1"])
    b.tools.db["flights"]["F3"]["flight_number"] = "F3"
    third = event(2, b, name=first.name, args=args)
    assert not detect_duplicate_queries([changed, third])[1]["duplicate_query"]


def test_flight_status_only_its_status_and_target_date_are_relevant():
    b = backend()
    args = {"flight_number": "F1", "date": "2024-05-15"}
    first = event(0, b, name="get_flight_status", args=args)
    b.tools.db["flights"]["F1"]["dates"]["2024-05-15"]["prices"]["economy"] = 100
    same = event(1, b, name=first.name, args=args)
    b.tools.db["flights"]["F1"]["dates"]["2024-05-15"]["status"] = "cancelled"
    changed = event(2, b, name=first.name, args=args)
    assert [x["duplicate_query"] for x in detect_duplicate_queries([first, same, changed])] == [
        False, True, False]


def test_missing_proof_never_equals_missing_proof_and_strict_fails_closed():
    events = [event(0), event(1)]
    assert all(x["status"] == "unresolved" for x in detect_duplicate_queries(events))
    with pytest.raises(MissingQueryEvidence, match="e0"):
        detect_duplicate_queries(events, require_evidence=True)


@pytest.mark.parametrize("field", ["db_hash_before", "db_hash_after", "query_state_before",
                                   "query_state_after", "query_state_scope", "query_state_version",
                                   "result"])
def test_strict_requires_complete_native_proof(field):
    e = event(0, backend())
    setattr(e, field, None)
    with pytest.raises(MissingQueryEvidence):
        detect_duplicate_queries([e], require_evidence=True)


def test_wrong_version_or_changed_read_state_is_unresolved():
    e = event(0, backend())
    e.query_state_version = "unknown"
    with pytest.raises(MissingQueryEvidence):
        detect_duplicate_queries([e], require_evidence=True)
    e = event(1, backend())
    e.query_state_after = "different"
    with pytest.raises(MissingQueryEvidence):
        detect_duplicate_queries([e], require_evidence=True)


def test_unexecuted_invalid_read_does_not_require_native_proof():
    e = event(0)
    e.error_kind = "schema_invalid"
    assert detect_duplicate_queries([e], require_evidence=True)[0]["status"] == "excluded"


def test_no_db_fixture_remains_importable_and_state_unknown_even_for_static_tools():
    assert capture_query_evidence(SimpleNamespace(), "get_user_details", {"user_id": "u"})[
        "query_state_before"] is None
    assert capture_query_evidence(SimpleNamespace(), "calculate", {"expression": "2+2"})[
        "query_state_before"] is None


def test_out_of_order_or_repeated_event_ids_rejected():
    b = backend()
    with pytest.raises(ValueError):
        detect_duplicate_queries([event(1, b), event(0, b)])
    with pytest.raises(ValueError):
        detect_duplicate_queries([event(0, b), event(0, b)])


def test_canonical_arguments_detects_reordered_json_keys_and_explicit_refresh():
    b = backend()
    args = {"origin": "SFO", "destination": "JFK", "date": "2024-05-15"}
    first = event(0, b, name="search_direct_flight", args=args)
    second = event(1, b, name=first.name, args=dict(reversed(list(args.items()))))
    second.user_requested_refresh = True
    assert detect_duplicate_queries([first, second], require_evidence=True)[1]["duplicate_query"]
    assert not second.duplicate_query  # Pure detection preserves the original.


def test_remarking_clears_stale_flags_and_strict_failure_is_atomic():
    b = backend()
    first, second = event(0, b), event(1, b)
    first.duplicate_query, first.duplicate_of_event_id = True, "stale"
    mark_duplicate_queries([first, second], require_evidence=True)
    assert not first.duplicate_query and first.duplicate_of_event_id is None
    first.duplicate_query, first.duplicate_of_event_id = True, "retained-on-failure"
    second.query_state_before = None
    with pytest.raises(MissingQueryEvidence):
        mark_duplicate_queries([first, second], require_evidence=True)
    assert first.duplicate_query and first.duplicate_of_event_id == "retained-on-failure"


@pytest.mark.parametrize("error_kind", ["mixed_content_and_tool_call", "multiple_tool_calls"])
def test_actor_rejected_named_reads_need_no_native_proof_and_preserve_errors(error_kind):
    from tau2_agentic_rl.reward.process_penalty import compute_process_penalty
    from tau2_agentic_rl.schemas import ToolEvent

    b = backend()
    rejected = [ToolEvent(
        event_id=f"e{i}", sequence=i, turn_id=i + 1, name="get_user_details",
        error_kind=error_kind, result="Locally rejected before native execution",
        duplicate_query=True, duplicate_of_event_id="stale",
    ) for i in (0, 2)]
    before = [item.model_dump(exclude={"duplicate_query", "duplicate_of_event_id"})
              for item in rejected]
    penalty = compute_process_penalty(rejected, assistant_turns=4)
    events = [rejected[0], event(1, b), rejected[1], event(3, b)]
    decisions = mark_duplicate_queries(events, require_evidence=True)
    assert [item["status"] for item in decisions] == [
        "excluded", "first_success", "excluded", "duplicate",
    ]
    assert decisions[3]["duplicate_of_event_id"] == "e1"
    assert all(not item.duplicate_query and item.duplicate_of_event_id is None
               for item in rejected)
    assert before == [item.model_dump(exclude={"duplicate_query", "duplicate_of_event_id"})
                      for item in rejected]
    assert compute_process_penalty(rejected, assistant_turns=4) == penalty
    assert [item["kind"] for item in penalty.events] == [error_kind, error_kind]


def test_onestop_deadend_changes_only_matter_when_a_complete_connection_appears():
    b = backend()
    flights = b.tools.db["flights"]
    flights["DEAD"] = deepcopy(flights["F1"])
    flights["DEAD"].update(flight_number="DEAD", destination="ORD")
    flights["NEXT"] = deepcopy(flights["F2"])
    flights["NEXT"].update(flight_number="NEXT", origin="ORD")
    flights["NEXT"]["dates"]["2024-05-15"]["status"] = "cancelled"
    args = {"origin": "SFO", "destination": "LAX", "date": "2024-05-15"}
    first = event(0, b, name="search_onestop_flight", args=args)
    flights["DEAD"]["dates"]["2024-05-15"]["prices"]["economy"] += 1
    flights["DEAD"]["dates"]["2024-05-15"]["available_seats"]["economy"] -= 1
    same = event(1, b, name=first.name, args=args)
    flights["NEXT"]["dates"]["2024-05-15"]["status"] = "available"
    changed = event(2, b, name=first.name, args=args)
    assert [item["duplicate_query"] for item in detect_duplicate_queries(
        [first, same, changed], require_evidence=True,
    )] == [False, True, False]

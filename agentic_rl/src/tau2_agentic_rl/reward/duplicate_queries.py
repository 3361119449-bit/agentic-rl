"""Deterministic, state-bound duplicate reads for pinned native Tau2 airline.

No semantic refresh exception exists: a successful identical query against the
same relevant state repeats already obtained information. Unknown state never
compares equal. The caller must reconstruct native proof before enabling v1.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from tau2_agentic_rl.versions import sha256_json

QUERY_STATE_VERSION = "duplicate-query-v1"
READ_ONLY_TOOLS = frozenset({
    "get_user_details", "get_reservation_details", "get_flight_status",
    "search_direct_flight", "search_onestop_flight", "list_all_airports", "calculate",
})
_UNEXECUTED_ERRORS = frozenset({
    "parse_error", "unknown_tool", "schema_invalid",
    "mixed_content_and_tool_call", "multiple_tool_calls",
})
_UNEXECUTED_STATUSES = frozenset({"blocked_by_invalid_sibling", "termination_control"})


class MissingQueryEvidence(ValueError):
    """Native state/result evidence must be reconstructed before v1 scoring."""


def canonical_arguments(arguments: Mapping[str, Any]) -> str:
    """Canonical JSON keys/whitespace only; preserve native string/list semantics.

    In particular IDs are case sensitive, and calculator whitespace may separate
    numeric literals. Trimming, case folding, or algebraic rewriting would merge
    calls that the native tools do not treat equivalently.
    """
    return json.dumps(
        dict(arguments), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _direct_candidates(flights: Mapping, *, origin: str, destination: str | None,
                       date: str, leave_after: str | None = None) -> list[dict]:
    """Project the exact date/routes traversed by native _search_direct_flight."""
    rows = []
    for flight in flights.values():
        # Filter on native scalar/model fields before serializing. Dumping
        # every date of every flight here multiplies one-stop capture cost by
        # the entire DB even though only a handful of route/date rows matter.
        if _get(flight, "origin") != origin:
            continue
        if destination is not None and _get(flight, "destination") != destination:
            continue
        instance = _get(flight, "dates", {}).get(date)
        if instance is None or _get(instance, "status") != "available":
            continue
        if leave_after is not None and _get(flight, "scheduled_departure_time_est") < leave_after:
            continue
        # Native result excludes every other date, and includes these fields.
        rows.append({
            key: _get(flight, key) for key in (
                "flight_number", "origin", "destination",
                "scheduled_departure_time_est", "scheduled_arrival_time_est",
            )
        } | {"date": date, "instance": _plain(instance)})
    return rows


def query_state_projection(backend: Any, name: str, arguments: Mapping) -> tuple[str, Any] | None:
    """Return (scope, relevant native DB projection), or None without proof.

    Only backend.tools.db is trusted, never the global hash or previous result.
    Flight search follows pinned Tau2's available-route/date and one-stop
    connection rules, including its literal next-day string behavior. Only
    complete one-stop connections contribute returned information.
    """
    if name not in READ_ONLY_TOOLS:
        return None
    db = _get(_get(backend, "tools"), "db")
    if db is None:
        return None
    args = dict(arguments)
    try:
        if name in {"get_user_details", "get_reservation_details"}:
            collection, key = (
                ("users", "user_id") if name == "get_user_details"
                else ("reservations", "reservation_id")
            )
            target = args[key]
            if not isinstance(target, str):
                return None
            records = _get(db, collection)
            if not isinstance(records, Mapping):
                return None
            return f"airline:{collection}:{target}", _plain(records.get(target))
        if name == "get_flight_status":
            flight_number, date = args["flight_number"], args["date"]
            flights = _get(db, "flights")
            if not isinstance(flights, Mapping):
                return None
            flight = flights.get(flight_number)
            instance = _get(flight, "dates", {}).get(date) if flight is not None else None
            # Native get_flight_status returns only this status, not price/seats.
            return f"airline:flight-status:{flight_number}:{date}", (
                {"status": _get(instance, "status")} if instance is not None else None
            )
        if name in {"search_direct_flight", "search_onestop_flight"}:
            flights = _get(db, "flights")
            if not isinstance(flights, Mapping):
                return None
            origin, destination, date = args["origin"], args["destination"], args["date"]
            if not all(isinstance(value, str) for value in (origin, destination, date)):
                return None
            scope = f"airline:{name}:{origin}:{destination}:{date}"
            if name == "search_direct_flight":
                return scope, _direct_candidates(
                    flights, origin=origin, destination=destination, date=date,
                )
            first_legs = _direct_candidates(
                flights, origin=origin, destination=None, date=date,
            )
            connections = []
            for first in first_legs:
                arrival = first["scheduled_arrival_time_est"]
                date2 = f"2024-05-{int(date[-2:]) + 1}" if "+1" in arrival else date
                second_legs = _direct_candidates(
                    flights, origin=first["destination"], destination=destination,
                    date=date2, leave_after=arrival,
                )
                # A dead-end first leg cannot affect the native result. Its
                # price/seats become relevant if a viable connection appears.
                if second_legs:
                    connections.append({"first": first, "second": second_legs})
            return scope, connections
        # Airports is READ; calculator is nonmutating GENERIC in pinned Tau2.
        # Requiring a native DB above distinguishes unavailable fixture evidence.
        return f"airline:static:{name}", {"native_semantics": QUERY_STATE_VERSION}
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def capture_query_evidence(backend: Any, name: str, arguments: Mapping) -> dict[str, str | None]:
    """Capture one instant; before/after are equal in this return value.

    Around a real call, combine **before with
    query_state_after=after["query_state_after"]. This same helper is used in the
    live adapter and frozen native replay. DB hashes and real result are captured
    separately at those exact call boundaries.
    """
    empty = dict.fromkeys((
        "query_state_before", "query_state_after", "query_state_scope", "query_state_version",
    ))
    projection = query_state_projection(backend, name, arguments)
    if projection is None:
        return empty
    scope, state = projection
    fingerprint = sha256_json({
        "version": QUERY_STATE_VERSION, "tool": name,
        "arguments": json.loads(canonical_arguments(arguments)), "scope": scope, "state": state,
    })
    return {
        "query_state_before": fingerprint, "query_state_after": fingerprint,
        "query_state_scope": scope, "query_state_version": QUERY_STATE_VERSION,
    }


def _missing_proof(event: Any) -> list[str]:
    missing = [
        field for field in (
            "db_hash_before", "db_hash_after", "query_state_before", "query_state_after",
            "query_state_scope", "query_state_version",
        )
        if not isinstance(_get(event, field), str) or not _get(event, field)
    ]
    if _get(event, "result") is None:
        missing.append("result")
    if _get(event, "query_state_version") != QUERY_STATE_VERSION:
        missing.append("supported_query_state_version")
    if _get(event, "query_state_before") != _get(event, "query_state_after"):
        missing.append("stable_read_state")
    return missing


def detect_duplicate_queries(events: Sequence[Any], *, require_evidence: bool = False) -> list[dict]:
    """Return auditable decisions in event order without changing outcomes.

    Only successful, proven reads seed the first-success index. Failed calls do
    not create or receive duplicate blame. Actor-local rejected tool turns
    and every write/control call are excluded. Strict mode raises before any
    caller mutation if an executed read lacks full native proof.
    """
    decisions = []
    seen: dict[tuple, str] = {}
    event_ids: set[str] = set()
    previous_sequence = -1
    for event in events:
        event_id, sequence = _get(event, "event_id"), _get(event, "sequence")
        if (
            not isinstance(event_id, str) or not event_id or event_id in event_ids
            or type(sequence) is not int or sequence <= previous_sequence
        ):
            raise ValueError("query events require unique IDs and increasing sequence")
        event_ids.add(event_id)
        previous_sequence = sequence
        name, args = _get(event, "name"), _get(event, "arguments", {})
        decision = {
            "event_id": event_id, "name": name, "duplicate_query": False,
            "duplicate_of_event_id": None, "status": "excluded",
            **{field: _get(event, field) for field in (
                "db_hash_before", "db_hash_after", "query_state_before", "query_state_after",
                "query_state_scope", "query_state_version",
            )},
        }
        decisions.append(decision)
        if (
            name not in READ_ONLY_TOOLS
            or _get(event, "error_kind") in _UNEXECUTED_ERRORS
            or _get(event, "execution_status") in _UNEXECUTED_STATUSES
        ):
            continue
        missing = _missing_proof(event)
        if missing:
            decision.update(status="unresolved", missing_evidence=missing)
            if require_evidence:
                raise MissingQueryEvidence(
                    f"duplicate-query-v1 event {event_id} needs native replay: {', '.join(missing)}"
                )
            continue
        if not _get(event, "success", False):
            decision["status"] = "failed_read"
            continue
        normalized = canonical_arguments(args)
        decision["canonical_arguments"] = normalized
        key = (
            name, normalized, _get(event, "query_state_scope"),
            _get(event, "query_state_version"), _get(event, "query_state_before"),
        )
        if key in seen:
            decision.update(
                status="duplicate", duplicate_query=True, duplicate_of_event_id=seen[key],
            )
        else:
            seen[key] = event_id
            decision["status"] = "first_success"
    return decisions


def mark_duplicate_queries(events: Sequence[Any], *, require_evidence: bool = False) -> list[dict]:
    """Recompute flags; preserve result/success/error fields and original events."""
    decisions = detect_duplicate_queries(events, require_evidence=require_evidence)
    for event, decision in zip(events, decisions, strict=True):
        for field in ("duplicate_query", "duplicate_of_event_id"):
            if isinstance(event, dict):
                event[field] = decision[field]
            else:
                setattr(event, field, decision[field])
    return decisions

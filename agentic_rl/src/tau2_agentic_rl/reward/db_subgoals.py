"""Optional, snapshot-verified reservation credit; never a scalar task reward."""

from __future__ import annotations

import json
import math

from tau2_agentic_rl.reward.required_actions import (
    MATCHER_VERSION,
    MUTATING_TOOLS,
    _event_satisfies,
    arguments_equal,
)
from tau2_agentic_rl.schemas import ToolEvent
from tau2_agentic_rl.versions import sha256_json

DB_SUBGOAL_VERSION = "db_subgoal_local_v1"
FIELDS = {
    "update_reservation_flights": ("cabin", "flights"),
    "update_reservation_passengers": ("passengers",),
    "update_reservation_baggages": ("total_baggages", "nonfree_baggages"),
}


def select_db_subgoals(task, required_actions, dependencies, transfer_rule):
    """Only independent updates of different fields on one existing order."""
    basis = (task.get("evaluation_criteria") or {}).get("reward_basis", [])
    writes = [a for a in required_actions if a["name"] in MUTATING_TOOLS]
    if (
        "DB" not in {str(v).upper() for v in basis}
        or transfer_rule.get("allowed")
        or transfer_rule.get("required")
        or task.get("user_tools")
        or dependencies
        or len(writes) < 2
        or any(a["name"] not in FIELDS for a in writes)
        or len({a["name"] for a in writes}) != len(writes)
        or len({a["arguments"].get("reservation_id") for a in writes}) != 1
        or not writes[0]["arguments"].get("reservation_id")
        or len({a["action_id"] for a in writes}) != len(writes)
    ):
        return []
    return writes


def _snapshot(value, reservation_id):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return None
    if not isinstance(value, dict) or value.get("reservation_id") != reservation_id:
        return None
    fields = {"reservation_id", "status", *(f for fs in FIELDS.values() for f in fs)}
    if not fields <= value.keys():
        return None
    if value["status"] not in (None, "cancelled"):
        return None
    for key in ("total_baggages", "nonfree_baggages"):
        if type(value[key]) is not int or value[key] < 0:
            return None
    for key, keys in (
        ("flights", ("flight_number", "date")),
        ("passengers", ("first_name", "last_name", "dob")),
    ):
        if (
            not isinstance(value[key], list)
            or not value[key]
            or any(
                not isinstance(item, dict)
                or any(not isinstance(item.get(k), str) or not item[k] for k in keys)
                for item in value[key]
            )
        ):
            return None
    return value


def _state_matches(action, snapshot):
    if snapshot is None or snapshot["status"] == "cancelled":
        return False
    fields = ("reservation_id", *FIELDS[action["name"]])
    expected = {key: action["arguments"].get(key) for key in fields}
    actual = {key: snapshot[key] for key in fields}
    return arguments_equal(expected, actual, tool_name=action["name"])


def build_db_subgoal_trace(
    *,
    task: dict,
    required_actions: list[dict],
    dependencies: list,
    transfer_rule: dict,
    initial_reservations: dict,
    initial_state_fingerprint: str,
    events: list[ToolEvent],
    violating_turns: list[int],
    turns: int,
) -> dict:
    """Process every write in sequence, but attribute only to its actor turn.

    Each goal can earn and lose at most 0.5/N once. Missing observations never
    prove regression. A later query cannot claim a previously withheld bonus.
    """
    if not initial_state_fingerprint or type(turns) is not int or turns < 0:
        raise ValueError("missing DB subgoal initial identity or turns")
    if (
        len({e.sequence for e in events}) != len(events)
        or any(not 1 <= e.turn_id <= turns for e in events)
        or [(e.turn_id, e.sequence) for e in events]
        != sorted((e.turn_id, e.sequence) for e in events)
    ):
        raise ValueError("DB subgoal events must be in original execution order")
    if any(type(t) is not int or not 0 <= t < turns for t in violating_turns):
        raise ValueError("invalid DB subgoal policy turns")
    goals = select_db_subgoals(task, required_actions, dependencies, transfer_rule)
    target = goals[0]["arguments"]["reservation_id"] if goals else None
    snapshot = _snapshot(initial_reservations.get(target), target) if goals else None
    # Initial state is required to distinguish new work from existing success.
    if snapshot is None:
        goals = []
    ids = [a["action_id"] for a in goals]
    initial = [_state_matches(a, snapshot) for a in goals]
    provenance, current = list(initial), list(initial)
    awarded, charged = set(), set()
    states, positives, negatives = [list(initial)], [], []
    wrong, deferred, regressions, unknown = [], [], [], []
    by_turn = {t: [] for t in range(1, turns + 1)}
    for e in events:
        by_turn[e.turn_id].append(e)
    for turn in range(1, turns + 1):
        candidates, lost = set(), set()
        blocked = False
        for e in by_turn[turn] if goals else []:
            if e.name not in MUTATING_TOOLS and e.db_effect is not True:
                continue
            semantic_event, contradictory_prior = e, False
            if e.name == "update_reservation_baggages" and e.state_before is not None:
                if snapshot is None:
                    # Without a trusted prior, require the matcher's strict
                    # fallback; never use an unverified count to waive payment.
                    semantic_event = e.model_copy(update={"state_before": None})
                else:
                    contradictory_prior = e.state_before != {
                        "reservation_id": target,
                        "nonfree_baggages": snapshot["nonfree_baggages"],
                    }
            matching = [
                i
                for i, a in enumerate(goals)
                if not contradictory_prior and _event_satisfies(a, semantic_event)
            ]
            if not matching:
                blocked = True
            if e.name == "send_certificate":
                # Pinned Tau2 changes the user's payment methods only. This
                # outside write blocks bonuses, but cannot erase order state.
                continue
            if e.db_effect is False:
                continue  # Proved no change: do not invent a state transition.
            affected_target = e.arguments.get("reservation_id")
            if affected_target and affected_target != target:
                continue
            before = list(current)
            trusted = (
                _snapshot(e.result, target)
                if e.success and e.db_effect is True and not contradictory_prior
                else None
            )
            if trusted is None:
                blocked = True
                current = [False] * len(goals)
                snapshot = None
                unknown.append(turn - 1)
                for i, a in enumerate(goals):
                    if e.name == a["name"]:
                        provenance[i] = False
                continue
            snapshot = trusted
            for i, a in enumerate(goals):
                if e.name == a["name"]:
                    provenance[i] = i in matching
                current[i] = provenance[i] and _state_matches(a, snapshot)
                if before[i] and not current[i]:
                    blocked = True
                    lost.add(i)
                if not before[i] and current[i] and i in matching:
                    candidates.add(i)
            if any(not current[i] for i in matching):
                # Matching inputs alone do not certify a correct write. A
                # contradictory result also blocks siblings' turn-level bonus.
                blocked = True
        eligible = {
            i for i in candidates if current[i] and not initial[i] and i not in awarded
        }
        if blocked:
            wrong.append(turn - 1)
        if eligible and (blocked or turn - 1 in violating_turns):
            deferred.append(turn - 1)
            eligible = set()
        if lost:
            regressions.append(turn - 1)
        charges = lost - charged
        awarded.update(eligible)
        charged.update(charges)
        positives.append([ids[i] for i in sorted(eligible)])
        negatives.append([ids[i] for i in sorted(charges)])
        states.append(list(current))
    definition = {
        "version": DB_SUBGOAL_VERSION,
        "matcher_version": MATCHER_VERSION,
        "task": sha256_json(task),
        "required_actions": sha256_json(required_actions),
        "dependencies": sha256_json(dependencies),
        "transfer_rule": sha256_json(transfer_rule),
        "initial_reservations": sha256_json(initial_reservations),
        "initial_state_fingerprint": initial_state_fingerprint,
        "goal_ids": ids,
    }
    weight = 0.5 / len(goals) if goals else 0.0
    trace = {
        "version": DB_SUBGOAL_VERSION,
        "applicable": bool(goals),
        "definition": definition,
        "definition_fingerprint": sha256_json(definition),
        "events_fingerprint": sha256_json([e.model_dump() for e in events]),
        "violating_turns": sorted(set(violating_turns)),
        "initial": initial,
        "check_values": states,
        "phi": [sum(s) / len(goals) if goals else 0.0 for s in states],
        "positive_goal_ids": positives,
        "negative_goal_ids": negatives,
        "turn_credits": [
            weight * (len(p) - len(n))
            for p, n in zip(positives, negatives, strict=True)
        ],
        "wrong_write_turns": wrong,
        "deferred_turns": deferred,
        "regression_turns": regressions,
        "unknown_state_turns": sorted(set(unknown)),
    }
    trace["trace_fingerprint"] = sha256_json(trace)
    validate_db_subgoal_trace(trace, turns=turns, violating_turns=violating_turns)
    return trace


def validate_db_subgoal_trace(trace, *, turns, violating_turns):
    if not isinstance(trace, dict):
        raise ValueError("missing DB subgoal trace")
    payload = {k: v for k, v in trace.items() if k != "trace_fingerprint"}
    definition = trace.get("definition", {})
    if (
        trace.get("version") != DB_SUBGOAL_VERSION
        or trace.get("trace_fingerprint") != sha256_json(payload)
        or trace.get("definition_fingerprint") != sha256_json(definition)
        or definition.get("matcher_version") != MATCHER_VERSION
        or trace.get("violating_turns") != sorted(set(violating_turns))
    ):
        raise ValueError("DB subgoal evidence identity changed")
    ids, initial = definition["goal_ids"], trace["initial"]
    values, phi = trace["check_values"], trace["phi"]
    if (
        len(set(ids)) != len(ids)
        or len(initial) != len(ids)
        or len(values) != turns + 1
        or len(phi) != turns + 1
        or values[0] != initial
        or trace["applicable"] != bool(ids)
        or any(
            len(s) != len(ids) or any(type(v) is not bool for v in s) for s in values
        )
        or any(
            not math.isclose(p, sum(s) / len(ids) if ids else 0.0, abs_tol=1e-12)
            for p, s in zip(phi, values, strict=True)
        )
    ):
        raise ValueError("invalid DB subgoal states")
    positive, negative, credits = (
        trace[k] for k in ("positive_goal_ids", "negative_goal_ids", "turn_credits")
    )
    if any(len(v) != turns for v in (positive, negative, credits)):
        raise ValueError("DB subgoal credit length differs from actor turns")
    seen, charged = set(), set()
    blocked = set(violating_turns) | set(trace["wrong_write_turns"])
    weight = 0.5 / len(ids) if ids else 0.0
    for t, (pos, neg, credit) in enumerate(
        zip(positive, negative, credits, strict=True)
    ):
        if (
            len(set(pos)) != len(pos)
            or len(set(neg)) != len(neg)
            or not set(pos + neg) <= set(ids)
            or seen.intersection(pos)
            or charged.intersection(neg)
            or (pos and t in blocked)
            or any(
                initial[ids.index(i)] or not values[t + 1][ids.index(i)] for i in pos
            )
            or type(credit) not in (float, int)
            or not math.isfinite(credit)
            or not math.isclose(credit, weight * (len(pos) - len(neg)), abs_tol=1e-12)
        ):
            raise ValueError("invalid or repeated DB subgoal local credit")
        seen.update(pos)
        charged.update(neg)
    return list(credits)


def subgoal_trace_from_scoring(inputs, events, turns, reward):
    """Shared by live scoring, frozen retries, record validation and audits."""
    project = inputs.get("reward_project_config", {})
    if project.get("credit", {}).get("db_subgoal_credit") is None:
        return None
    from tau2_agentic_rl.advantages import CreditConfig

    CreditConfig.from_project(project)
    frozen = inputs.get("db_subgoal_inputs")
    if not isinstance(frozen, dict):
        raise ValueError("missing frozen DB subgoal inputs")
    return build_db_subgoal_trace(
        task=inputs["judge"]["task"],
        required_actions=inputs["required_actions"],
        dependencies=inputs["action_dependencies"],
        transfer_rule=inputs["judge"]["transfer_rule"],
        events=events,
        turns=turns,
        violating_turns=reward.details["policy_credit"]["violating_turns"],
        **frozen,
    )

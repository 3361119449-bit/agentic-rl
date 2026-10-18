"""Preserve first official completion credit and limit destructive-turn credit."""

from __future__ import annotations

import math

from tau2_agentic_rl.reward.db_subgoals import validate_db_subgoal_trace
from tau2_agentic_rl.reward.progress import validate_progress_trace
from tau2_agentic_rl.reward.required_actions import MUTATING_TOOLS, arguments_equal
from tau2_agentic_rl.versions import sha256_json

POST_COMPLETION_VERSION = "post_completion_regression_v1"
COMPLETION_ADVANTAGE_MODE = "positive_floor_v1"
COMPLETION_POSITIVE_MARGIN = 0.5
MIN_POSITIVE_ADVANTAGE = 1e-6
EXTERNAL_FAILURES = {
    "infrastructure_error",
    "infrastructure_failure",
    "unexpected_error",
    "timeout",
    "user_error",
}


def build_post_completion_trace(
    *,
    task,
    progress_trace,
    events,
    turns,
    required_actions,
    transfer_rule,
    violating_turns,
    termination_reason,
    subgoal_trace=None,
):
    """Only official, applicable checks prove completion; phi is never a proxy.

    Turn identifiers in this trace are one-based. Token mappings and the input
    policy attribution remain zero-based, as in existing ProCredit v4.
    """
    validate_progress_trace(progress_trace, turns=turns)
    if (
        [(e.turn_id, e.sequence) for e in events]
        != sorted((e.turn_id, e.sequence) for e in events)
        or len({e.sequence for e in events}) != len(events)
        or any(not 1 <= e.turn_id <= turns for e in events)
    ):
        raise ValueError("completion events differ from original turn order")
    if any(type(t) is not int or not 0 <= t < turns for t in violating_turns):
        raise ValueError("invalid completion policy turn")
    if subgoal_trace is not None:
        validate_db_subgoal_trace(
            subgoal_trace, turns=turns, violating_turns=violating_turns
        )
    # Positive goal awards and negative regressions can share a turn. Count
    # gross awards, not net local credit, when sharing the positive budget.
    db_positive = (
        sum(len(p) for p in subgoal_trace["positive_goal_ids"])
        * (0.5 / len(subgoal_trace["initial"]) if subgoal_trace["initial"] else 0)
        if subgoal_trace is not None
        else 0.0
    )
    criteria = task.get("evaluation_criteria") or {}
    basis = {str(b).upper() for b in criteria.get("reward_basis", [])}
    skip = None
    if "DB" not in basis or not basis <= {"DB", "COMMUNICATE"}:
        skip = "unsupported_official_reward_basis"
    elif (
        task.get("user_tools")
        or transfer_rule.get("allowed")
        or transfer_rule.get("required")
    ):
        skip = "unsupported_external_or_transfer_state"
    elif termination_reason in EXTERNAL_FAILURES:
        skip = "external_failure"
    names = [c["check_id"] for c in progress_trace["checks"]]
    required = ["db"] + (
        [f"comm:{i}" for i in range(len(criteria.get("communicate_info") or []))]
        if "COMMUNICATE" in basis
        else []
    )
    if not set(required) <= set(names):
        skip = skip or "missing_official_prefix_checks"
    if not skip and progress_trace["definition"].get("task") != sha256_json(task):
        raise ValueError("completion task differs from verified official prefixes")
    official = [False] * (turns + 1)
    db = [False] * (turns + 1)
    if not skip:
        official = [
            all(s[names.index(c)] for c in required)
            for s in progress_trace["check_values"]
        ]
        db = [s[names.index("db")] for s in progress_trace["check_values"]]
    initial = official[0]
    first = (
        next((t for t in range(1, turns + 1) if official[t]), None)
        if not initial
        else None
    )
    regressions, unattributed, causes = [], [], {}
    seen = initial
    for t in range(1, turns + 1):
        if seen and db[t - 1] and not db[t]:
            writes = [
                e
                for e in events
                if e.turn_id == t
                and e.name in MUTATING_TOOLS
                and e.db_effect is True
                and e.execution_status is None
            ]
            if (
                writes
                and progress_trace["prefix_lengths"][t]
                > progress_trace["prefix_lengths"][t - 1]
            ):
                regressions.append(t)
                causes[str(t)] = [e.event_id for e in writes]
            else:
                unattributed.append(t)
        seen = seen or official[t]
    bonus = [0.0] * turns
    safe = first is not None and first - 1 not in violating_turns
    if first is not None:
        for e in events:
            if e.turn_id != first or (
                e.name not in MUTATING_TOOLS and e.db_effect is not True
            ):
                continue
            if (
                not e.success
                or e.error_kind
                or not any(
                    e.name == a["name"]
                    and arguments_equal(
                        a["arguments"],
                        e.arguments,
                        tool_name=e.name,
                        state_before=e.state_before,
                    )
                    for a in required_actions
                )
            ):
                safe = False
    if regressions and safe:
        bonus[first - 1] = max(0.0, 0.5 - db_positive)
    definition = {
        "version": POST_COMPLETION_VERSION,
        "completion_advantage_mode": COMPLETION_ADVANTAGE_MODE,
        "task": sha256_json(task),
        "progress_trace": sha256_json(progress_trace),
        "events": sha256_json([e.model_dump() for e in events]),
        "required_actions": sha256_json(required_actions),
        "transfer_rule": sha256_json(transfer_rule),
        "termination_reason": termination_reason,
        "violating_turn_indices": sorted(set(violating_turns)),
        "subgoal_trace": subgoal_trace["trace_fingerprint"]
        if subgoal_trace is not None
        else None,
    }
    result = {
        "version": POST_COMPLETION_VERSION,
        "turn_index_base": 1,
        "definition": definition,
        "applicable": skip is None,
        "skip_reason": skip,
        "official_check_ids": required,
        "official_success": official,
        "db_success": db,
        "initial_success": initial,
        "first_success_turn": first,
        "first_success_safe": safe,
        "positive_floor_turn": first if regressions and safe else None,
        "positive_margin": COMPLETION_POSITIVE_MARGIN,
        "min_positive_advantage": MIN_POSITIVE_ADVANTAGE,
        "regression_turns": regressions,
        "unattributed_db_drop_turns": unattributed,
        "causal_event_ids": causes,
        "turn_bonus": bonus,
        "penalty_threshold": 1.0,
        "db_positive_credit": db_positive,
        "completion_positive_credit": math.fsum(bonus),
    }
    result["trace_fingerprint"] = sha256_json(result)
    validate_post_completion_trace(
        result,
        turns=turns,
        violating_turns=violating_turns,
        subgoal_trace=subgoal_trace,
    )
    return result


def validate_post_completion_trace(
    trace, *, turns, violating_turns, subgoal_trace=None
):
    if not isinstance(trace, dict):
        raise ValueError("missing post-completion trace")
    payload = {k: v for k, v in trace.items() if k != "trace_fingerprint"}
    definition = trace.get("definition", {})
    if (
        trace.get("version") != POST_COMPLETION_VERSION
        or definition.get("completion_advantage_mode") != COMPLETION_ADVANTAGE_MODE
        or trace.get("turn_index_base") != 1
        or trace.get("trace_fingerprint") != sha256_json(payload)
        or definition.get("violating_turn_indices") != sorted(set(violating_turns))
        or definition.get("subgoal_trace")
        != (
            subgoal_trace.get("trace_fingerprint")
            if subgoal_trace is not None
            else None
        )
        or trace.get("penalty_threshold") != 1.0
        or trace.get("positive_margin") != COMPLETION_POSITIVE_MARGIN
        or trace.get("min_positive_advantage") != MIN_POSITIVE_ADVANTAGE
    ):
        raise ValueError("post-completion evidence identity changed")
    values, db, bonus = (
        trace[k] for k in ("official_success", "db_success", "turn_bonus")
    )
    regression = trace["regression_turns"]
    first = trace["first_success_turn"]
    safe = trace["first_success_safe"]
    floor_turn = trace["positive_floor_turn"]
    db_positive = (
        sum(len(p) for p in subgoal_trace["positive_goal_ids"])
        * (0.5 / len(subgoal_trace["initial"]) if subgoal_trace["initial"] else 0)
        if subgoal_trace is not None
        else 0.0
    )
    if (
        len(values) != turns + 1
        or len(db) != turns + 1
        or len(bonus) != turns
        or any(type(v) is not bool for v in values + db)
        or any(
            success and not db_success
            for success, db_success in zip(values, db, strict=True)
        )
        or type(trace["initial_success"]) is not bool
        or trace["initial_success"] != values[0]
        or first
        != (
            next((t for t in range(1, turns + 1) if values[t]), None)
            if not values[0]
            else None
        )
        or type(safe) is not bool
        or (safe and (first is None or first - 1 in violating_turns))
        or (floor_turn is not None and type(floor_turn) is not int)
        or floor_turn != (first if regression and safe else None)
        or regression != sorted(set(regression))
        or any(
            type(t) is not int
            or not 1 <= t <= turns
            or not db[t - 1]
            or db[t]
            or not any(values[:t])
            or not trace["causal_event_ids"].get(str(t))
            for t in regression
        )
        or any(
            type(v) not in (int, float)
            or not math.isfinite(v)
            or not 0 <= v <= 0.5
            or (v > 0 and (floor_turn != i + 1 or i in violating_turns))
            for i, v in enumerate(bonus)
        )
        or not math.isclose(trace["db_positive_credit"], db_positive, abs_tol=1e-12)
        or not math.isclose(
            trace["completion_positive_credit"], math.fsum(bonus), abs_tol=1e-12
        )
        or db_positive + math.fsum(bonus) > 0.5 + 1e-12
        or (
            not trace["applicable"]
            and (regression or any(bonus) or floor_turn is not None)
        )
    ):
        raise ValueError("invalid or duplicated post-completion credit")
    return list(bonus), {t - 1 for t in regression}


def completion_trace_from_scoring(inputs, events, turns, reward, subgoal_trace=None):
    project = inputs.get("reward_project_config", {})
    if project.get("credit", {}).get("post_completion_credit") is None:
        return None
    from tau2_agentic_rl.advantages import CreditConfig

    CreditConfig.from_project(project)
    return build_post_completion_trace(
        task=inputs["judge"]["task"],
        progress_trace=inputs["progress_trace"],
        events=events,
        turns=turns,
        required_actions=inputs["required_actions"],
        transfer_rule=inputs["judge"]["transfer_rule"],
        violating_turns=reward.details["policy_credit"]["violating_turns"],
        termination_reason=inputs["judge"]["trajectory"]["termination_reason"],
        subgoal_trace=subgoal_trace,
    )

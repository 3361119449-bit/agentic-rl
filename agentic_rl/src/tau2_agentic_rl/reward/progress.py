"""Verify immutable delivered prefixes; never replay actions in the live world."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Callable

from tau2_agentic_rl.advantages import validate_phi
from tau2_agentic_rl.environment.replay import rejection_aware_constructor
from tau2_agentic_rl.reward.required_actions import (
    MATCHER_VERSION,
    evaluate_required_actions,
)
from tau2_agentic_rl.reward.transfer_branch import (
    _communication_score,
    _pre_transfer_score,
    successful_transfer_event,
)
from tau2_agentic_rl.schemas import ToolEvent
from tau2_agentic_rl.versions import sha256_json

PROGRESS_VERSION = "progress-v1"
PROGRESS_DEDUP_VERSION = "progress-v2"


def evaluate_tau2_prefix(task: dict, messages: list[dict]) -> dict:
    """Run the pinned official verifiers on newly constructed environments.

    Imports are lazy so scoring math and saved trace replay do not require Tau2.
    EnvironmentEvaluator constructs predicted/gold worlds itself; no live
    adapter, user simulator, or LLM is passed to it.
    """
    from pydantic import TypeAdapter
    from tau2.data_model.message import Message
    from tau2.data_model.tasks import Task
    from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator
    from tau2.registry import registry

    parsed_task = Task.model_validate(task)
    parsed_messages = TypeAdapter(list[Message]).validate_python(messages)
    criteria = task.get("evaluation_criteria") or {}
    basis = {str(value).upper() for value in criteria.get("reward_basis", [])}
    db = None
    if "DB" in basis:
        result = EnvironmentEvaluator.calculate_reward(
            environment_constructor=rejection_aware_constructor(registry.get_env_constructor("airline")),
            task=parsed_task,
            full_trajectory=parsed_messages,
            solo_mode=False,
            strict_replay=True,
        )
        if result.db_check is None:
            raise ValueError("official prefix evaluator omitted applicable DB check")
        db = result.db_check.db_reward == 1.0
    communication = CommunicateEvaluator.calculate_reward(
        task=parsed_task, full_trajectory=parsed_messages
    )
    return {
        "db": db,
        "communicate": [
            check.met for check in (communication.communicate_checks or [])
        ],
    }


def build_progress_trace(
    *,
    task: dict,
    messages: list[dict],
    prefix_lengths: list[int],
    events: list[ToolEvent],
    required_actions: list[dict],
    dependencies: list[list[str]],
    transfer_rule: dict,
    initial_state_fingerprint: str,
    evaluator: Callable[[dict, list[dict]], dict] | None = None,
    version: str = PROGRESS_VERSION,
) -> dict:
    """One initial prefix plus one prefix per generated assistant turn."""
    if version not in {PROGRESS_VERSION, PROGRESS_DEDUP_VERSION}:
        raise ValueError("unknown progress version")
    if (
        not prefix_lengths
        or any(type(n) is not int or n < 0 or n > len(messages) for n in prefix_lengths)
        or prefix_lengths != sorted(prefix_lengths)
        or prefix_lengths[-1] != len(messages)
    ):
        raise ValueError("invalid delivered progress prefix boundaries")
    if not initial_state_fingerprint:
        raise ValueError("missing initial state fingerprint")
    if transfer_rule.get("allowed") and not transfer_rule.get("required"):
        raise ValueError("ambiguous optional transfer progress target")
    transfer = bool(transfer_rule.get("required"))
    if transfer and not transfer_rule.get("allowed"):
        raise ValueError("required transfer must be allowed")
    turns = len(prefix_lengths) - 1
    if any(event.turn_id < 1 or event.turn_id > turns for event in events):
        raise ValueError("tool event lies outside recorded assistant turns")
    evaluator = evaluator or evaluate_tau2_prefix
    cached, states, ids = {}, [], None
    communication = (task.get("evaluation_criteria") or {}).get(
        "communicate_info"
    ) or []
    for turn, length in enumerate(prefix_lengths):
        prefix = messages[:length]
        prefix_events = [event for event in events if event.turn_id <= turn]
        state = {}
        if transfer:
            transfer_event = successful_transfer_event(prefix_events)
            state["transfer_call"] = transfer_event is not None
            for index, group in enumerate(
                transfer_rule.get("required_pre_transfer_action_groups", [])
            ):
                state[f"transfer_pre:{index}"] = (
                    _pre_transfer_score(prefix_events, [group], transfer_event) == 1
                )
            for index, check in enumerate(
                transfer_rule.get("required_communication_checks", [])
            ):
                state[f"transfer_comm:{index}"] = (
                    _communication_score(prefix, [check]) == 1
                )
        else:
            if length not in cached:
                cached[length] = evaluator(deepcopy(task), deepcopy(prefix))
            official = cached[length]
            if official["db"] is not None:
                state["db"] = official["db"]
            if len(official["communicate"]) != len(communication):
                raise ValueError("official progress COMM checks are incomplete")
            for index, value in enumerate(official["communicate"]):
                state[f"comm:{index}"] = value
            completed = evaluate_required_actions(
                required_actions, prefix_events, dependencies
            ).completed_action_ids
            for action in required_actions:
                check_id = f"req:{action['action_id']}"
                if check_id in state:
                    raise ValueError("duplicate required-action check")
                state[check_id] = str(action["action_id"]) in completed
        if any(type(value) is not bool for value in state.values()):
            raise ValueError("progress verifiers must return binary checks")
        if ids is None:
            ids = list(state)
        if list(state) != ids:
            raise ValueError("progress check applicability changed during trajectory")
        states.append(list(state.values()))
    checks = [
        {"check_id": key, "initial": states[0][index], "included": (
            not states[0][index] and not (
                version == PROGRESS_DEDUP_VERSION and "db" in ids and key.startswith("req:")
            )
        )}
        for index, key in enumerate(ids)
    ]
    included = [index for index, item in enumerate(checks) if item["included"]]
    definition = {
        "version": version,
        "task": sha256_json(task),
        "required_actions": sha256_json(required_actions),
        "dependencies": sha256_json(dependencies),
        "transfer_rule": sha256_json(transfer_rule),
        "checks": checks,
    }
    if version == PROGRESS_DEDUP_VERSION:
        definition["matcher_version"] = MATCHER_VERSION
    result = {
        "version": version,
        "definition": definition,
        "checks": checks,
        "checkset_fingerprint": sha256_json(definition),
        "initial_state_fingerprint": initial_state_fingerprint,
        "prefix_lengths": list(prefix_lengths),
        "check_values": states,
        "phi": [
            sum(state[index] for index in included) / len(included) if included else 0.0
            for state in states
        ],
    }
    validate_progress_trace(result, turns=turns)
    return result


def validate_progress_trace(trace: dict, *, turns: int) -> list[float]:
    """Reject invented/misaligned progress, including offline tampering."""
    phi = validate_phi(trace.get("phi"), turns)
    version = trace.get("version")
    if version not in {PROGRESS_VERSION, PROGRESS_DEDUP_VERSION}:
        raise ValueError("unknown progress version")
    definition, checks = trace.get("definition", {}), trace.get("checks", [])
    if definition.get("version") != version or (
        version == PROGRESS_DEDUP_VERSION and definition.get("matcher_version") != MATCHER_VERSION
    ):
        raise ValueError("progress definition version changed")
    if (
        trace.get("checkset_fingerprint") != sha256_json(definition)
        or definition.get("checks") != checks
        or not trace.get("initial_state_fingerprint")
    ):
        raise ValueError("progress checkset identity mismatch")
    if len({item["check_id"] for item in checks}) != len(checks):
        raise ValueError("duplicate progress check ID")
    has_db = any(item["check_id"] == "db" for item in checks)
    if any(
        type(item["initial"]) is not bool
        or type(item["included"]) is not bool
        or item["included"] != (
            not item["initial"] and not (
                version == PROGRESS_DEDUP_VERSION and has_db and item["check_id"].startswith("req:")
            )
        )
        for item in checks
    ):
        raise ValueError("invalid initial progress applicability")
    values = trace.get("check_values", [])
    if len(values) != turns + 1 or any(
        len(state) != len(checks) or any(type(v) is not bool for v in state)
        for state in values
    ):
        raise ValueError("incomplete progress checks")
    if values[0] != [item["initial"] for item in checks]:
        raise ValueError("initial progress checks changed")
    active = [index for index, item in enumerate(checks) if item["included"]]
    for actual, state in zip(phi, values, strict=True):
        expected = sum(state[index] for index in active) / len(active) if active else 0
        if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError("progress fraction differs from verified checks")
    return phi


def validate_progress_inputs(
    trace: dict,
    *,
    required_actions: list,
    dependencies: list,
    transfer_rule: dict,
) -> None:
    for name, value in {
        "required_actions": required_actions,
        "dependencies": dependencies,
        "transfer_rule": transfer_rule,
    }.items():
        if trace.get("definition", {}).get(name) != sha256_json(value):
            raise ValueError(f"progress {name} changed; regenerate verified prefixes")

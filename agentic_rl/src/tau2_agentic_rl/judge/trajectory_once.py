"""One cached request returns trajectory checks and all causal turn decisions."""

import json
from copy import deepcopy
from dataclasses import replace
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictInt, model_validator

from tau2_agentic_rl.judge.client import DeepSeekJudge
from tau2_agentic_rl.judge.prompts import build_judge_messages
from tau2_agentic_rl.judge.simple_turn import check_label_consistency
from tau2_agentic_rl.reward.simple_turn import (
    INDEPENDENT_VERSION,
    RefusalRequest,
    TurnDecision,
    build_label_trace,
    turn_request,
    validate_decision,
)
from tau2_agentic_rl.schemas import JudgeResult

MODE = "trajectory_once_v1"
PROMPT_VERSION = "trajectory-once-causal-v3-message-index"
SCORER_CODE_VERSION = "trajectory-once-v4-message-index"

# Preserve the previously tested whole-trajectory rubric and full receipt rules.
TURN_PROMPT = """
Also return turns for every requested target. For its message_index i, judge
only visible_messages[i]; use visible_messages[0..i] and the bound Policy.
Task/reference facts, global tool-event outcomes and future messages are not
evidence available to that generation. Global findings do not excuse or taint
a different turn. Cite supporting zero-based visible message indices. The target
message_index already identifies the current generation; include it in evidence
only if useful.

For a tool call at i, judge its name, arguments and authorization using facts
available before the call. Its own receipt is always out of scope. Do not state
or infer that this call succeeded, failed, returned nothing, or changed anything.
A past call's result can support a reason only as the result of that past call.
For a later assistant reply, cite the earlier receipt if the reply reports its outcome.
Later events cannot justify or invalidate an earlier decision.

Before emitting each row, keep only its cited messages and the Policy. If a factual
premise or the verdict loses support, reassess using eligible evidence. If a fact
cannot be established from the prefix, treat it as unknown. Return the final JSON.

bad: any current error, even in recovery or a refusal. Check claims, capabilities,
Policy, arguments and latest intent. Incompletion alone is insufficient. Search
must match dates, airports, direction and constraints. Cabin-only: check booked
segments/dates; a direct search cannot verify connecting segments. Wrong-target
reads are bad; extra reads of real retained segments can be neutral. Judge misuse
at the quote. Distinguish allowed cabin-only changes from basic-economy flight
changes. Verify status per flight before ruling out cancellation. Check transfer
summary facts separately.

Money: recompute from user/tool facts, counts, fees and retained original fares.
Distinguish total from difference, quote from settlement and charge from refund.
Match reported settlements to actual earlier payment entries. Rank eligible fares.
Before writes check details, consent and foreseeable charge against authorization.
A clear instruction can confirm where Policy permits. A corrected authorized
action stands on its own. Ambiguous lookup wording alone does not prove fabrication.

refusal: an explicitly requested prohibited action, grounded facts and fully
correct explanation/alternatives. Evaluate each refusal independently, including
repeats. Any error wins. Otherwise neutral. Use task_guidance when relevant.

Follow the single combined rubric.output_schema.
Every target message_index appears exactly once; no extra fields. verdict is bad,
refusal or neutral. refusal_requests lists {"target":"object","operation":"action"}
only for refusal, otherwise []. Quote 16-80 literal characters from raw_generated_text
if present, else content/tool call (whole text if shorter).
Native trajectory evidence IDs are ONLY for trajectory_result. For turns, use
message_index exactly as listed in targets and cite visible_messages indices.
Native messages with visible_message_index reference that visible message's content.
"""


class BatchTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    turn_index: StrictInt | None = None  # Compatibility with previously frozen replies.
    message_index: StrictInt
    target_quote: str
    verdict: Literal["bad", "refusal", "neutral"]
    evidence_message_indices: list[StrictInt]
    reason: str
    refusal_requests: list[RefusalRequest]


class TrajectoryOnceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trajectory_result: JudgeResult
    turns: list[BatchTurn]

    @model_validator(mode="before")
    @classmethod
    def normalize_flat_wrapper(cls, value):
        """Accept known envelope variants without changing any verdicts."""
        if isinstance(value, dict) and value.get("type") == "json_object":
            value = {k: v for k, v in value.items() if k != "type"}
        required = {"semantic_checks", "transfer_semantic_checks",
                    "mandatory_policy_checks", "transfer_check", "turns"}
        if isinstance(value, dict) and required <= value.keys() <= required | {"schema_version"}:
            return {"trajectory_result": {k: v for k, v in value.items() if k != "turns"},
                    "turns": value["turns"]}
        return value


def decisions_from_result(result, turn_inputs):
    if turn_inputs.get("version") != INDEPENDENT_VERSION:
        raise ValueError("trajectory_once_v1 requires independent simple-turn-v2")
    targets = turn_inputs["turn_message_indices"]
    returned = {row.message_index: row for row in result.turns}
    if len(returned) != len(result.turns) or returned.keys() != set(targets):
        missing = sorted(set(targets) - returned.keys())
        unexpected = sorted(returned.keys() - set(targets))
        raise ValueError(
            "Return each target message_index exactly once. "
            f"Missing {missing}; unexpected {unexpected}; "
            f"duplicate rows {len(result.turns) - len(returned)}."
        )
    decisions, errors = [], []
    for t, index in enumerate(targets):
        row = returned[index]
        request = turn_request(turn_inputs, t, [])
        # The required message_index and literal quote still bind the label
        # to this generation. Repeating the target in evidence is redundant.
        evidence = list(dict.fromkeys(row.evidence_message_indices))
        if index not in evidence:
            evidence.append(index)
        try:
            decision = TurnDecision(
                turn_index=t if row.turn_index is None else row.turn_index,
                target_message_index=index,
                target_generation_sha256=request["target_generation_sha256"],
                target_quote=row.target_quote, bad_turn=row.verdict == "bad",
                correct_refusal=row.verdict == "refusal", reason=row.reason,
                evidence_message_indices=evidence,
                refusal_requests=row.refusal_requests, revoked_refusals=[],
            )
            validated = validate_decision(decision.model_dump(), request)
            check_label_consistency(decision)
            decisions.append(validated)
        except ValueError as error:
            errors.append(f"message_index {index} (cite 0..{index}): {error}")
    if errors:
        raise ValueError("Invalid turn rows. " + "; ".join(errors))
    return decisions


def build_messages(*, turn_inputs, **global_inputs):
    if turn_inputs.get("version") != INDEPENDENT_VERSION:
        raise ValueError("trajectory_once_v1 requires simple-turn-v2")
    if turn_inputs["policy"] != global_inputs["policy"]:
        raise ValueError("global and turn Judge policies differ")
    visible = []
    targets = []
    for i, original in enumerate(turn_inputs["messages"]):
        message = deepcopy(original)
        message.pop("turn_idx", None)
        message.pop("assistant_turn_id", None)
        if message.get("role") == "system" and isinstance(message.get("content"), str):
            message["content"] = message["content"].replace(
                turn_inputs["policy"], "[See FIXED_AIRLINE_POLICY.]")
        visible.append({"message_index": i, "message": message})
    for t, i in enumerate(turn_inputs["turn_message_indices"]):
        # Verify frozen target shape before any HTTP request.
        turn_request(turn_inputs, t, [])
        targets.append({"message_index": i})
    if not targets:
        raise ValueError("trajectory_once_v1 requires at least one target")

    # Shared delivered content is transmitted once. Preserve native evidence IDs
    # and unmatched native entries; never replace the actor-visible view with
    # environment-only facts or results of rejected/undelivered actions.
    trajectory = deepcopy(global_inputs["trajectory"])

    def signature(message):
        return json.dumps({k: message.get(k) for k in ("role", "content", "tool_calls")},
                          ensure_ascii=False, sort_keys=True)

    positions = {}
    for item in visible:
        positions.setdefault(signature(item["message"]), []).append(item["message_index"])
    used = set()
    compact = []
    for native in trajectory.get("messages", []):
        matches = positions.get(signature(native), [])
        index = next((i for i in matches if i not in used), None)
        if index is None:
            compact.append(native)
            continue
        used.add(index)
        compact.append({k: v for k, v in native.items() if k not in {"content", "tool_calls"}}
                       | {"visible_message_index": index})
    trajectory["messages"] = compact
    messages = build_judge_messages(
        **{**global_inputs, "trajectory": trajectory},
        turn_output_schema={
            "message_index": targets[0]["message_index"], "target_quote": "literal target excerpt",
            "verdict": "bad|refusal|neutral", "evidence_message_indices": [],
            "reason": "short supported reason", "refusal_requests": [],
        },
    )
    messages[0]["content"] += TURN_PROMPT
    messages[1]["content"] += "\n\nACTOR_VISIBLE_TIMELINE_AND_TARGETS:\n" + json.dumps(
        {"visible_messages": visible, "targets": targets}, ensure_ascii=False)
    return messages


class TrajectoryOnceJudge(DeepSeekJudge):
    result_model = TrajectoryOnceResult
    build_messages = staticmethod(build_messages)

    @staticmethod
    def _validate_requested_criteria(result, inputs):
        DeepSeekJudge._validate_requested_criteria(result.trajectory_result, inputs)
        decisions_from_result(result, inputs["turn_inputs"])


async def evaluate_scoring_inputs(judge, inputs, *, progress=None):
    """Shared by live rollout and frozen rescoring; no fallback between modes."""
    mode = inputs.get("reward_project_config", {}).get("judge", {}).get("mode", "per_turn")
    if mode == "per_turn":
        return await judge.evaluate(**inputs["judge"])
    if mode != MODE:
        raise ValueError(f"unknown judge.mode: {mode}")
    config = replace(
        judge.config, cache_dir=str(judge.cache_dir / MODE),
        prompt_version=PROMPT_VERSION, rubric_version=PROMPT_VERSION,
        scorer_code_version=SCORER_CODE_VERSION,
        timeout_seconds=max(judge.config.timeout_seconds, 750),
        thinking_type="enabled", reasoning_effort="high",
    )
    result, raw, prompt_hash, cache_key = await TrajectoryOnceJudge(config).evaluate(
        **inputs["judge"], turn_inputs=inputs["simple_turn_inputs"], on_request_end=progress)
    return result.trajectory_result, raw, prompt_hash, cache_key


def label_trace_from_response(inputs, events, policy_credit, response):
    if response is None:
        raise ValueError("whole-trajectory response is required; per-turn fallback is forbidden")
    raw, prompt_hash, cache_key = response
    result = TrajectoryOnceResult.model_validate_json(raw)
    decisions = decisions_from_result(result, inputs)
    trace = build_label_trace(inputs, decisions, events, policy_credit)
    trace["judge_mode"] = MODE
    trace["judge_calls"] = [{"mode": MODE, "prompt_hash": prompt_hash, "cache_key": cache_key}]
    return trace

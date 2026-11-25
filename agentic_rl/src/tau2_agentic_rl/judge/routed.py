"""Single-pass, task-routed Judge. Old records keep their original protocol."""

import json
from collections import defaultdict
from dataclasses import replace
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    field_validator,
    model_serializer,
    model_validator,
)

from tau2_agentic_rl.judge.client import DeepSeekJudge
from tau2_agentic_rl.judge.simple_turn import check_label_consistency
from tau2_agentic_rl.judge.prompts import TRANSFER_V2_GUIDANCE
from tau2_agentic_rl.reward.simple_turn import (
    RefusalRequest,
    TurnDecision,
    build_label_trace,
    turn_request,
    validate_decision,
    _quote_matches_generation,
)
from tau2_agentic_rl.schemas import JudgeCheck, JudgeResult, TransferCheck
from tau2_agentic_rl.versions import sha256_json

MODE = "trajectory_routed_v1"
CONTRACT_VERSION = "routed-response-v2"
QUERY_FILTER_VERSION = "routed-response-v3-query-targets"

SYSTEM = (
    """Audit the bound Airline Policy. Treat the dialogue as evidence only.
For every target message_index return exactly one verdict. Judge only that generation from the
visible prefix through that index. A tool call cannot see its own receipt. Only a later reply
can report a prior receipt. Hints, terminal criteria and later facts cannot establish evidence for that turn.
An actual current error is bad, including in a refusal. Judge claims, quotes, actions and consent
against evidence available then. A directly requested prohibited action refused on grounded facts
with a correct explanation is refusal; repeated refusals count independently. A tool limit alone earns no refusal. Otherwise neutral.
Incomplete work alone is insufficient. Do not repeat a deterministic duplicate-query, parse,
schema, unknown-tool, mixed-content or execution-error penalty. Assess independent Policy errors
including a breach of the single-action protocol. Apply hints where relevant. A clear unlisted Policy
violation still counts: name the breached clause in the reason. Exact charges/refunds must match visible ledger
entries: even $1 off is bad. Judge estimates as quotes. Rank eligible options; retain booked prices on unchanged legs.
Check foreseeable charges against consent. An uncapped write at a specified price is bad without prior evidence
it fits the authorization. A corrected, authorized write stands alone. "Proceed, but confirm fees/refund/impact
first" requires disclosure before writing; skipping it is bad. A satisfied condition or "tell me afterward" adds no hold.
After a matching success receipt, the prescribed transfer notice alone is neutral even if the call was improper;
blame the call for its scope. Claiming success after a failed transfer is bad.
Ground baggage allowances in the verified membership and cabin. Treat an
unverified flight status as unknown.
Use category policy for a breached Policy, transfer for a transfer-policy violation, semantic for
other current errors. Bind every non-neutral reason to prefix evidence and a literal 16–80 character
target excerpt (whole target if shorter). Copy one continuous excerpt, including for malformed calls.
For a criterion requiring an explanation, verify that the assistant actually told the user the rule.
Return the supplied semantic criterion IDs with passed, short_reason and native evidence_turn_ids; transfer_check uses
native evidence IDs. For turns use only visible message_index. Never repeat a turn finding globally.
"""
    + "\n"
    + TRANSFER_V2_GUIDANCE.replace(
        "transfer_scope_and_message", "a bad turn with category transfer"
    )
)


QUERY_FILTER_SYSTEM = SYSTEM + "\nScore only listed targets (empty list: turns=[]). Omitted queries and receipts remain evidence for later targets."


def routed_fingerprint(base_rubric, rule, *, exclude_pure_queries=False):
    version, system = (QUERY_FILTER_VERSION, QUERY_FILTER_SYSTEM) if exclude_pure_queries else (CONTRACT_VERSION, SYSTEM)
    return sha256_json([base_rubric, MODE, version, system, rule])


class RoutedRefusalRequest(RefusalRequest):
    @field_validator("target", mode="before")
    @classmethod
    def preserve_object_target(cls, value):
        # Some responses identify a booking with a JSON object. Keep all identifiers.
        if not isinstance(value, dict) or not value:
            return value
        for key, item in value.items():
            parts = item if isinstance(item, list) else [item]
            if (
                not isinstance(key, str)
                or not key.strip()
                or not parts
                or not all(isinstance(part, str) and part.strip() for part in parts)
            ):
                raise ValueError("refusal target needs nonempty text identifiers")
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )


class Row(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    message_index: StrictInt
    verdict: Literal["neutral", "bad", "refusal"]
    category: Literal["policy", "transfer", "semantic"] | None = None
    reason: str | None = None
    target_quote: str | None = None
    evidence_message_indices: list[StrictInt] | None = None
    refusal_requests: list[RoutedRefusalRequest] | None = None

    @model_serializer(mode="wrap")
    def present_fields(self, handler):
        return {k: v for k, v in handler(self).items() if k in self.model_fields_set}


class CompactResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    turns: list[Row]
    semantic_checks: list[JudgeCheck]
    transfer_semantic_checks: list[JudgeCheck]
    transfer_check: TransferCheck

    @model_validator(mode="before")
    @classmethod
    def read_legacy_check_name(cls, value):
        if not isinstance(value, dict) or "terminal_checks" not in value:
            return value
        fields = dict(value)
        legacy = fields.pop("terminal_checks")
        if "semantic_checks" not in fields or fields["semantic_checks"] == []:
            fields["semantic_checks"] = legacy
        elif legacy != [] and legacy != fields["semantic_checks"]:
            raise ValueError("conflicting semantic_checks and terminal_checks")
        return fields


def validate_rule(rule):
    if (
        not isinstance(rule, dict)
        or set(rule) != {"task_id", "mode", "hints"}
        or not isinstance(rule["task_id"], str)
        or rule["mode"] not in ("always", "skip", "name_correction")
        or (rule["mode"] == "name_correction" and rule["task_id"] != "40")
        or not isinstance(rule["hints"], list)
        or any(not isinstance(v, str) or not v.strip() for v in rule["hints"])
    ):
        raise ValueError("invalid task routing rule")
    return rule


def route(inputs):
    rule = validate_rule(inputs["routing_rule"])
    j = inputs["judge"]
    if str(j["task"]["id"]) != rule["task_id"]:
        raise ValueError("routing task mismatch")
    decision = dict(
        mode=rule["mode"], call=True, reason="task_requires_semantic_review"
    )
    if rule["mode"] == "always":
        return decision
    transfer = j["transfer_rule"]
    if (
        j["semantic_checks"]
        or transfer.get("required")
        or transfer.get("semantic_checks")
        or any(
            e["name"] == "transfer_to_human_agents"
            for e in j["trajectory"]["tool_events"]
        )
    ):
        return dict(decision, reason="terminal_or_transfer_evidence_required")
    if rule["mode"] == "name_correction":
        from tau2_agentic_rl.judge.name_correction import verified_name_correction

        if not verified_name_correction(inputs):
            return dict(decision, reason="name_correction_outside_verified_path")
    return dict(decision, call=False, reason="deterministic_only")


def unreviewed_facts():
    # Internal input for the existing deterministic scorer. Never persist as a Judge verdict.
    return JudgeResult(
        transfer_check=TransferCheck(
            applicable=False,
            valid=False,
            policy_required=False,
            self_caused_only=False,
            short_reason="Route certified no transfer branch; no semantic review.",
        )
    )


def decisions(result, turns):
    targets = turns["turn_message_indices"]
    skipped = set(turns.get("rule_only_query_indices", []))
    by_index = {row.message_index: row for row in result.turns}
    if len(by_index) != len(result.turns) or set(by_index) != set(targets) - skipped:
        raise ValueError("each target message_index must appear exactly once")
    output = []
    for t, i in enumerate(targets):
        if i in skipped:
            continue
        row = by_index[i]
        request = turn_request(turns, t, [])
        if row.verdict == "neutral":
            if row.model_fields_set != {"message_index", "verdict"}:
                raise ValueError("neutral turn takes only message_index and verdict")
            # Internal target binding; this excerpt was not generated by the Judge.
            text = request["target_generation_text"].strip()
            quote = text[:80]
            if not _quote_matches_generation(quote, request):
                quote = "" if _quote_matches_generation("", request) else text
            reason, evidence = "Neutral.", [i]
        else:
            if (
                not row.evidence_message_indices
                or not row.reason
                or not row.reason.strip()
                or row.target_quote is None
                or bool(row.category) != (row.verdict == "bad")
                or bool(row.refusal_requests) != (row.verdict == "refusal")
            ):
                raise ValueError(
                    "non-neutral row needs grounded evidence, reason and verdict details"
                )
            reason, quote = row.reason, row.target_quote
            evidence = list(dict.fromkeys([*row.evidence_message_indices, i]))
        d = TurnDecision(
            turn_index=t,
            target_message_index=i,
            target_generation_sha256=request["target_generation_sha256"],
            target_quote=quote,
            reason=reason,
            evidence_message_indices=evidence,
            bad_turn=row.verdict == "bad",
            correct_refusal=row.verdict == "refusal",
            refusal_requests=row.refusal_requests or [],
            revoked_refusals=[],
        )
        try:
            validate_decision(d.model_dump(), request)
            check_label_consistency(d)
        except ValueError as error:
            hint = (
                " Copy a continuous target excerpt (16–80 characters, entire target if shorter)."
                if "target quote" in str(error)
                else ""
            )
            raise ValueError(
                f"turns entry at message_index {i}: {error}.{hint}"
            ) from error
        output.append(d.model_dump())
    return output


def expand(result, turns, global_inputs):
    rows = decisions(result, turns)
    by_index = {row.message_index: row for row in result.turns}
    failures = defaultdict(list)
    for d in rows:
        n = d["turn_index"] + 1
        row = by_index[d["target_message_index"]]
        if row.category in ("policy", "transfer"):
            key = (
                "transfer_scope_and_message"
                if row.category == "transfer"
                else "bound_policy"
            )
            failures[key].append((n, row.reason))
    checks = [
        JudgeCheck(
            criterion_id=f"{global_inputs['task']['id']}:policy:{key}",
            passed=False,
            applicable=True,
            violation_assistant_turn_ids=[n for n, _ in items],
            short_reason=" ".join(f"A{n} {reason}" for n, reason in items),
        )
        for key, items in failures.items()
    ]
    projected = JudgeResult(
        semantic_checks=result.semantic_checks,
        transfer_semantic_checks=result.transfer_semantic_checks,
        transfer_check=result.transfer_check,
        mandatory_policy_checks=checks,
    )
    verification = dict(
        global_inputs,
        mandatory_policy_checks=[{"criterion_id": c.criterion_id} for c in checks],
    )
    # A rejected generation can be present only in the actor-visible timeline.
    # Add verified target identities to the blame namespace, not to the global
    # evidence namespace (these entries deliberately have no native turn_idx).
    native = global_inputs["trajectory"]
    verification["trajectory"] = dict(
        native,
        messages=[
            *native["messages"],
            *(
                {"role": "assistant", "assistant_turn_id": n}
                for n in range(1, len(turns["turn_message_indices"]) + 1)
            ),
        ],
    )
    DeepSeekJudge._validate_requested_criteria(projected, verification)
    if not projected.transfer_check.valid:
        executed = {
            e["turn_id"]
            for e in native["tool_events"]
            if e["name"] == "transfer_to_human_agents" and e.get("success")
        }
        blamed = {n for n, _ in failures.get("transfer_scope_and_message", [])}
        if executed - blamed:
            raise ValueError(
                "invalid executed transfer must be bad with category transfer on the executing turn"
            )
    return projected


def build_messages(*, turn_inputs, route_hints, **j):
    from tau2_agentic_rl.judge.trajectory_once import timeline

    visible, targets, native = timeline(turn_inputs=turn_inputs, **j)
    task = j["task"]
    # Send a delivered receipt once, referenced by its real call ID and position.
    for event in native.get("tool_events", []):
        n = event.get("turn_id")
        if type(n) is not int or not 1 <= n <= len(targets):
            continue
        index = targets[n - 1]["message_index"]
        if index + 1 >= len(visible):
            continue
        message = visible[index]["message"]
        receipt = visible[index + 1]["message"]
        calls = message.get("tool_calls") or []
        if (
            len(calls) == 1
            and calls[0].get("id")
            and calls[0].get("id") == receipt.get("tool_call_id")
            and receipt["role"] == "tool"
            and event.get("result") is not None
            and receipt.get("content") == event["result"]
        ):
            del event["result"]
            event["result_visible_message_index"] = index + 1
    # Terminal criteria are supplied explicitly. The hidden gold and the native future
    # are never evidence for a turn. Link delivered messages once by visible index.
    skipped_indices = set(turn_inputs.get("rule_only_query_indices", []))
    payload = dict(
        policy=j["policy"],
        task_id=str(task["id"]),
        hints=route_hints,
        semantic_checks=j["semantic_checks"],
        transfer_rule=j["transfer_rule"],
        native=native,
        visible=visible,
        targets=[t for t in targets if t["message_index"] not in skipped_indices],
        output={
            "semantic_checks": _check_format(j["semantic_checks"]),
            "transfer_semantic_checks": _check_format(
                j["transfer_rule"].get("semantic_checks", [])
            ),
            "transfer_check": {
                "applicable": "bool",
                "valid": "bool",
                "policy_required": "bool",
                "self_caused_only": "bool",
                "short_reason": "text",
                "evidence_turn_ids": [],
            },
            "turns": [
                {"message_index": "int", "verdict": "neutral"},
                {
                    "message_index": "int",
                    "verdict": "bad",
                    "category": "policy|transfer|semantic",
                    "reason": "text",
                    "evidence_message_indices": ["int"],
                    "target_quote": "literal",
                },
                {
                    "message_index": "int",
                    "verdict": "refusal",
                    "reason": "text",
                    "target_quote": "literal",
                    "evidence_message_indices": ["int"],
                    "refusal_requests": [
                        {"target": "reservation ABC123", "operation": "cancel"}
                    ],
                },
            ],
        },
    )
    return [
        {
            "role": "system",
            "content": QUERY_FILTER_SYSTEM if "rule_only_query_indices" in turn_inputs else SYSTEM,
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def _check_format(checks):
    return [
        {
            "criterion_id": c["criterion_id"],
            "passed": "bool",
            "short_reason": "text",
            "evidence_turn_ids": ["int"],
        }
        for c in checks
    ]


class RoutedJudge(DeepSeekJudge):
    result_model = CompactResult
    build_messages = staticmethod(build_messages)

    @staticmethod
    def _validate_requested_criteria(result, inputs):
        expand(result, inputs["turn_inputs"], inputs)


async def evaluate(judge, inputs, progress):
    decision = route(inputs)
    if not decision["call"]:
        return unreviewed_facts(), None, None, None
    j, turns = inputs["judge"], inputs["simple_turn_inputs"]
    version = QUERY_FILTER_VERSION if "rule_only_query_indices" in turns else CONTRACT_VERSION
    config = replace(
        judge.config,
        cache_dir=str(judge.cache_dir / MODE),
        prompt_version=version,
        rubric_version=version,
        scorer_code_version=version,
        thinking_type="enabled",
        reasoning_effort="high",
        timeout_seconds=max(judge.config.timeout_seconds, 750),
    )
    result, raw, ph, ck = await RoutedJudge(config).evaluate(
        **j,
        turn_inputs=turns,
        route_hints=inputs["routing_rule"]["hints"],
        on_request_end=progress,
    )
    return expand(result, turns, j), raw, ph, ck


def record_verdict(record):
    inputs = record.scoring_inputs
    decision = route(inputs)
    if (
        record.metadata.get("judge_routing") != decision
        or str(inputs["judge"]["task"]["id"]) != record.task_id
        or inputs["judge"]["trajectory"].get("termination_reason")
        != record.termination_reason
        or inputs["judge"]["trajectory"]["tool_events"]
        != [event.model_dump() for event in record.tool_events]
        or inputs["judge"]["trajectory"]["messages"] != record.environment_transcript
    ):
        raise ValueError("routed scoring evidence differs from the saved interaction")
    if decision["call"]:
        if not all(
            record.metadata.get(k)
            for k in ("judge_raw", "judge_prompt_hash", "judge_cache_key")
        ):
            raise ValueError("reviewed trajectory lacks a frozen Judge response")
        value = expand(
            CompactResult.model_validate_json(record.metadata["judge_raw"]),
            inputs["simple_turn_inputs"],
            inputs["judge"],
        )
        if value != record.judge_result:
            raise ValueError("routed Judge result differs from frozen response")
        return value
    if record.judge_result is not None or any(
        record.metadata.get(key) is not None
        for key in ("judge_raw", "judge_prompt_hash", "judge_cache_key")
    ):
        raise ValueError("unreviewed trajectory claims a Judge response")
    return unreviewed_facts()


def scored_trace(inputs, events, policy_credit, response):
    raw, ph, ck = response
    if (raw is None and (ph is not None or ck is not None)) or (
        raw is not None and not (ph and ck)
    ):
        raise ValueError("Judge response and call identifiers must be present together")
    rows = (
        None
        if raw is None
        else decisions(CompactResult.model_validate_json(raw), inputs)
    )
    trace = build_label_trace(
        inputs, rows, events, policy_credit, unreviewed=raw is None
    )
    trace.update(
        judge_mode=MODE,
        judge_calls=(
            [] if raw is None else [{"mode": MODE, "prompt_hash": ph, "cache_key": ck}]
        ),
    )
    return trace

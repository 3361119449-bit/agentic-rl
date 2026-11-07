"""Versioned binary turn labels and the final, uncentered advantage rule."""

from copy import deepcopy
import json

from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt

from tau2_agentic_rl.versions import sha256_json

VERSION = "simple-turn-v1"
JUDGE_VERSION = "simple-turn-target-v2"
TOOL_ERRORS = frozenset({
    "parse_error", "unknown_tool", "schema_invalid",
    "model_caused_execution_error", "mixed_content_and_tool_call",
})


class RefusalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    target: str
    operation: str


class RefusalRevocation(RefusalRequest):
    turn_index: StrictInt


class TurnDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    turn_index: StrictInt
    target_message_index: StrictInt
    target_generation_sha256: str
    target_quote: str
    bad_turn: StrictBool
    correct_refusal: StrictBool
    reason: str
    evidence_message_indices: list[StrictInt]
    refusal_requests: list[RefusalRequest]
    revoked_refusals: list[RefusalRevocation]


def request_key(request):
    key = (request["target"].strip().casefold(), request["operation"].strip().casefold())
    if not all(key):
        raise ValueError("correct refusal requires target and operation")
    return key


def enabled(project):
    rule = project.get("credit", {}).get("final_turn_rule")
    if rule not in (None, VERSION):
        raise ValueError("unknown final turn rule")
    return rule == VERSION


def transform_advantage(a0, bad_turn, eligible_correct_refusal):
    if bad_turn:
        return min(a0, -1.0)
    if eligible_correct_refusal:
        return max(a0 + 1.0, 1.0)
    return a0


def build_turn_inputs(messages, policy, turns):
    """Freeze exactly the bounded messages, never the environment transcript."""
    if type(turns) is not int or turns < 1 or not isinstance(policy, str) or not policy.strip():
        raise ValueError("simple turn labels require turns and bound policy")
    indices = []
    for index, message in enumerate(messages):
        if message.get("role") == "assistant" and "turn_idx" in message:
            if type(message["turn_idx"]) is not int or message["turn_idx"] != len(indices) + 1:
                raise ValueError("bounded assistant generation order changed")
            indices.append(index)
    if len(indices) != turns:
        raise ValueError("bounded messages differ from assistant turns")
    return {"version": VERSION, "messages": deepcopy(messages), "policy": policy,
            "turn_message_indices": indices}


def _judge_message(message):
    view = deepcopy(message)
    # These original actor/native IDs have different bases/namespaces. Keep
    # them in frozen inputs, but never ask the Judge to resolve their meaning.
    view.pop("turn_idx", None)
    view.pop("assistant_turn_id", None)
    return view


def _generation_text(message):
    raw = message.get("raw_generated_text")
    if isinstance(raw, str) and raw.strip():
        return raw
    parts = []
    if isinstance(message.get("content"), str) and message["content"].strip():
        parts.append(message["content"])
    if message.get("tool_calls"):
        parts.append(json.dumps(message["tool_calls"], ensure_ascii=False, sort_keys=True))
    return "\n".join(parts) or json.dumps(message, ensure_ascii=False, sort_keys=True)


def _target_hash(current):
    return sha256_json({"judge_version": JUDGE_VERSION, "current_generation": current})


def validate_target_request(request):
    target = request.get("target_message_index")
    current = request.get("current_generation")
    history = request.get("history")
    if (request.get("request_version") != JUDGE_VERSION
        or type(target) is not int or target < 0
        or not isinstance(current, dict) or current.get("message_index") != target
        or not isinstance(history, list) or len(history) != target
        or [item.get("message_index") for item in history] != list(range(target))
        or current.get("message", {}).get("role") != "assistant"):
        raise ValueError("invalid explicit current-generation target request")
    if any(key in item.get("message", {})
           for item in history + [current] for key in ("turn_idx", "assistant_turn_id")):
        raise ValueError("Judge request contains conflicting actor turn metadata")
    if (request.get("target_generation_sha256") != _target_hash(current)
        or request.get("target_generation_text") != _generation_text(current["message"])):
        raise ValueError("current-generation target fingerprint or text changed")
    return target


def turn_request(inputs, turn_index, previous_decisions):
    end = inputs["turn_message_indices"][turn_index]
    prior = [{"turn_index": d["turn_index"], "refusal_requests": deepcopy(d["refusal_requests"])}
             for d in previous_decisions if d["correct_refusal"] and not d["bad_turn"]]
    current = {"message_index": end, "message": _judge_message(inputs["messages"][end])}
    request = {"request_version": JUDGE_VERSION, "turn_index": turn_index,
               "target_message_index": end, "current_generation": current,
               "target_generation_sha256": _target_hash(current),
               "target_generation_text": _generation_text(current["message"]),
               "history": [{"message_index": i, "message": _judge_message(m)}
                           for i, m in enumerate(inputs["messages"][:end])],
               "policy": inputs["policy"], "previous_refusals": prior}
    validate_target_request(request)
    return request


def validate_decision(decision, request):
    target = validate_target_request(request)
    d = TurnDecision.model_validate(decision).model_dump()
    if d["turn_index"] != request["turn_index"] or not d["reason"].strip():
        raise ValueError("turn label identity or reason missing")
    text, quote = request["target_generation_text"], d["target_quote"]
    if (d["target_message_index"] != target
        or d["target_generation_sha256"] != request["target_generation_sha256"]
        or not quote.strip() or quote not in text
        or len(quote.strip()) < min(16, len(text.strip())) or len(quote) > 160):
        raise ValueError("decision target identity/quote must match the current generation")
    evidence = d["evidence_message_indices"]
    if (len(set(evidence)) != len(evidence)
        or any(i < 0 or i > target for i in evidence)
        or target not in evidence):
        raise ValueError("turn label evidence must include the current generation within the causal prefix")
    keys = [request_key(r) for r in d["refusal_requests"]]
    if bool(keys) != d["correct_refusal"] or len(set(keys)) != len(keys):
        raise ValueError("refusal requests must be distinct and match correct_refusal")
    revoked = [(r["turn_index"], request_key(r)) for r in d["revoked_refusals"]]
    prior = {(p["turn_index"], request_key(r)) for p in request["previous_refusals"]
             for r in p["refusal_requests"]}
    if (len(set(revoked)) != len(revoked) or set(revoked) - prior
        or any(i >= d["turn_index"] for i, _ in revoked) or (revoked and not d["bad_turn"])):
        raise ValueError("refusal revocation requires an earlier refusal and current illegal commitment")
    if revoked and target not in evidence:
        raise ValueError("revocation requires current generation evidence")
    return d


def build_label_trace(inputs, decisions, events, policy_credit):
    turns = len(inputs["turn_message_indices"])
    if inputs != build_turn_inputs(inputs["messages"], inputs["policy"], turns):
        raise ValueError("invalid frozen turn label inputs")
    if len(decisions) != turns:
        raise ValueError("missing turn labels")
    if not policy_credit.get("attribution_complete"):
        raise ValueError("simple turn labels require complete old policy attribution")
    policy_bad = policy_credit["violating_turns"]
    if any(type(t) is not int or not 0 <= t < turns for t in policy_bad):
        raise ValueError("invalid policy label turn")
    evidence = [[] for _ in range(turns)]
    for t in policy_bad:
        evidence[t].append({"source": "policy_credit", "reason": "existing policy violation"})
    for event in events:
        event = event.model_dump() if hasattr(event, "model_dump") else event
        if event.get("error_kind") not in TOOL_ERRORS:
            continue
        turn_id = event.get("turn_id")
        if type(turn_id) is not int or not 1 <= turn_id <= turns:
            raise ValueError("tool error lacks an attributable generation")
        evidence[turn_id - 1].append({"source": "tool_error", "event_id": event["event_id"],
                                     "reason": event["error_kind"]})
    parsed, labels, seen, revoked_by = [], [], set(), {}
    for t, value in enumerate(decisions):
        d = validate_decision(value, turn_request(inputs, t, parsed))
        parsed.append(d)
        bad = bool(evidence[t]) or d["bad_turn"]
        evidence[t].append({"source": "semantic", "reason": d["reason"],
                            "message_indices": d["evidence_message_indices"]})
        keys = [request_key(r) for r in d["refusal_requests"]]
        award_keys = [key for key in keys if key not in seen and not bad]
        if not bad:
            seen.update(keys)
        for revocation in d["revoked_refusals"]:
            # Resolve the request itself, including references through a
            # repeated refusal whose award was vetoed by a tool/policy error.
            revoked_by[request_key(revocation)] = t
        labels.append({"turn_index": t, "bad_turn": bad,
                       "dedup_keys": [list(key) for key in keys],
                       "eligible_refusal_keys": [list(key) for key in award_keys],
                       "evidence": evidence[t]})
    for label in labels:
        revoked_keys = [key for key in label["dedup_keys"]
                        if tuple(key) in revoked_by and revoked_by[tuple(key)] > label["turn_index"]]
        label["revoked_requests"] = [
            {"dedup_key": key, "revoked_by_turn": revoked_by[tuple(key)]} for key in revoked_keys
        ]
        label["eligible_refusal_keys"] = [key for key in label["eligible_refusal_keys"] if key not in revoked_keys]
        label["eligible_correct_refusal"] = bool(label["eligible_refusal_keys"])
        label["refusal_bonus"] = int(label["eligible_correct_refusal"])
    return {"version": VERSION, "judge_version": JUDGE_VERSION, "inputs_sha256": sha256_json(inputs),
            "decisions": parsed, "turns": labels}


def validate_labels(trace, turns):
    if not isinstance(trace, dict) or trace.get("version") != VERSION:
        raise ValueError("missing simple turn labels")
    if trace.get("judge_version") != JUDGE_VERSION:
        raise ValueError("legacy target-unbound Judge labels require frozen rescoring")
    labels = trace.get("turns")
    if not isinstance(labels, list) or len(labels) != turns:
        raise ValueError("turn label coverage differs from actor turns")
    for t, label in enumerate(labels):
        if (type(label.get("turn_index")) is not int or label["turn_index"] != t
            or type(label.get("bad_turn")) is not bool
            or type(label.get("eligible_correct_refusal")) is not bool
            or (label["bad_turn"] and label["eligible_correct_refusal"])):
            raise ValueError("invalid simple turn labels")
    return labels


def validate_record_labels(record):
    inputs = record.scoring_inputs
    if sha256_json(inputs) != record.metadata.get("scoring_inputs_sha256"):
        raise ValueError("frozen simple turn scoring inputs changed")
    expected = build_turn_inputs(record.messages, inputs["judge"]["policy"], record.assistant_turns)
    if inputs.get("simple_turn_inputs") != expected:
        raise ValueError("bounded record messages differ from frozen simple turn inputs")
    trace = record.metadata.get("simple_turn_labels")
    validate_labels(trace, record.assistant_turns)
    rebuilt = build_label_trace(expected, trace["decisions"], record.tool_events,
                                record.custom_reward.details["policy_credit"])
    if any(trace.get(k) != v for k, v in rebuilt.items()):
        raise ValueError("simple turn labels differ from frozen evidence")
    return trace

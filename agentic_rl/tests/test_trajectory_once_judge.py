"""No network: exercise the actual cached HTTP transport and label consumers."""

import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from tau2_agentic_rl.config import load_yaml, validate_procredit_config
from tau2_agentic_rl.failures import JudgeServiceFailure
from tau2_agentic_rl.judge.client import DeepSeekJudge, JudgeConfig
from tau2_agentic_rl.judge.simple_turn import score_simple_turns
from tau2_agentic_rl.judge.trajectory_once import (
    MODE,
    TrajectoryOnceJudge,
    TrajectoryOnceResult,
    build_messages,
    decisions_from_result,
    evaluate_scoring_inputs,
)
from tau2_agentic_rl.reward.simple_turn import (
    build_label_trace,
    build_turn_inputs,
    turn_request,
)


def frozen_inputs():
    messages = [
        {"role": "system", "content": "Bound policy."},
        {"role": "user", "content": "Please look up my booking."},
        {"role": "assistant", "content": "I will check your booking now.", "turn_idx": 1},
        {"role": "tool", "content": "Booking A costs $120."},
        {"role": "assistant", "content": "The booking price is $120.", "turn_idx": 2},
    ]
    native = deepcopy(messages[1:])
    for i, m in enumerate(native):
        if m["role"] == "assistant":
            m["assistant_turn_id"] = m["turn_idx"]
        m["turn_idx"] = i
    return {
        "reward_project_config": {"judge": {"mode": MODE}},
        "judge": {
            "task": {}, "policy": "Bound policy.",
            "trajectory": {"messages": native, "tool_events": []},
            "semantic_checks": [], "mandatory_policy_checks": [], "transfer_rule": {},
        },
        "simple_turn_inputs": build_turn_inputs(messages, "Bound policy.", 2, version="simple-turn-v2"),
    }


def response_for(inputs):
    turns = []
    for t in range(len(inputs["simple_turn_inputs"]["turn_message_indices"])):
        req = turn_request(inputs["simple_turn_inputs"], t, [])
        turns.append({
            "turn_index": t, "message_index": req["target_message_index"],
            "target_quote": req["target_generation_text"][:60], "verdict": "neutral",
            "reason": "The current claim is supported by the visible prefix.",
            "evidence_message_indices": [req["target_message_index"]], "refusal_requests": [],
        })
    return {"trajectory_result": {"semantic_checks": [], "transfer_semantic_checks": [],
                                  "mandatory_policy_checks": [], "transfer_check": {}},
            "turns": turns}


def mock_client(monkeypatch, scratch_dir, responder, *, max_retries=0):
    monkeypatch.setenv("TEST_JUDGE_KEY", "offline-dummy-key")
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(
        transport=httpx.MockTransport(responder), **kw))
    return DeepSeekJudge(JudgeConfig(model="fixture", api_key_env="TEST_JUDGE_KEY",
                                    cache_dir=str(scratch_dir), max_retries=max_retries))


def test_one_http_request_returns_global_and_all_turns_and_cache_reuses(monkeypatch, scratch_dir):
    inputs = frozen_inputs()
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        text = json.dumps(payload["messages"])
        assert text.count("Bound policy.") == 1
        assert text.count("I will check your booking now.") == 1
        assert payload["thinking"] == {"type": "enabled"}
        assert payload["reasoning_effort"] == "high"
        assert "max_tokens" not in payload
        return httpx.Response(200, json={"choices": [{"message": {
            "content": json.dumps(response_for(inputs))}}]})

    judge = mock_client(monkeypatch, scratch_dir, respond)

    async def run():
        result, raw, prompt_hash, cache_key = await evaluate_scoring_inputs(judge, inputs)
        trace = await score_simple_turns(
            inputs["simple_turn_inputs"], [], {"violating_turns": [0], "attribution_complete": True},
            judge, mode=MODE, batch_response=(raw, prompt_hash, cache_key))
        assert result.semantic_checks == []
        assert len(trace["turns"]) == 2 and len(trace["judge_calls"]) == 1
        assert trace["turns"][0]["bad_turn"]  # Global policy priority is unchanged.
        return trace

    first = asyncio.run(run())
    assert len(calls) == 1
    assert asyncio.run(run()) == first
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("raw", "tool_call", "quote", "bad"),
    [
        ("**Checked bags:** 3 (all free, as a gold member)", False,
         "Checked bags: 3 (all free, as a gold member)", False),
        ('<tool_call>{"name":"search_direct_flight","arguments":'
         '{"origin":"ATL","destination":"EWR","date":"2024-05-24"}}</tool_call>',
         True, '{"destination": "EWR", "date": "2024-05-24"}', False),
        ('<tool_call>{"name":"transfer_to_human_agents","arguments":'
         '{"summary":"The customer has no travel insurance on file."}}</tool_call>',
         True, '"no travel insurance on file"', False),
        ("\u7528\u6237<|im_end|>", False, "\u7528\u6237", True),
        ("<|im_end|>", False, "", True),
        ("", False, "", True),
    ],
)
def test_target_quote_tolerates_only_display_formatting(raw, tool_call, quote, bad):
    frozen = frozen_inputs()["simple_turn_inputs"]
    message = frozen["messages"][2]
    message.update(raw_generated_text=raw, content="" if tool_call else raw.removesuffix("<|im_end|>"))
    if tool_call:
        message["tool_calls"] = [json.loads(raw.removeprefix("<tool_call>").removesuffix("</tool_call>"))]
    answer = response_for({"simple_turn_inputs": frozen})
    answer["turns"][0].update(target_quote=quote, verdict="bad" if bad else "neutral")
    decisions = decisions_from_result(TrajectoryOnceResult.model_validate(answer), frozen)
    trace = build_label_trace(frozen, decisions, [], {
        "violating_turns": [], "attribution_complete": True,
    })
    assert trace["decisions"][0]["target_quote"] == quote
    assert trace["turns"][0]["bad_turn"] is bad


@pytest.mark.parametrize(
    "quote",
    ['{"date": "2024-05-25"}', '{"total_baggages": 2}', '"total_baggages":2', ""],
)
def test_tool_excerpt_still_rejects_changed_arguments_or_missing_quote(quote):
    frozen = frozen_inputs()["simple_turn_inputs"]
    raw = '{"name":"example","arguments":{"date":"2024-05-24","total_baggages":20}}'
    frozen["messages"][2].update(raw_generated_text=raw, content="", tool_calls=[json.loads(raw)])
    answer = response_for({"simple_turn_inputs": frozen})
    answer["turns"][0]["target_quote"] = quote
    with pytest.raises(ValueError, match="current generation"):
        decisions_from_result(TrajectoryOnceResult.model_validate(answer), frozen)


@pytest.mark.parametrize(
    "fault",
    ["future_receipt", "missing", "duplicate",
     "wrong_target", "wrong_index", "contradiction", "bad_refusal"],
)
def test_reject_bad_bulk_outputs_without_fallback(monkeypatch, scratch_dir, fault):
    inputs = frozen_inputs()
    answer = response_for(inputs)
    row = answer["turns"][0]
    if fault == "future_receipt":
        row["evidence_message_indices"].append(3)
    elif fault == "missing":
        answer["turns"].pop()
    elif fault == "duplicate":
        answer["turns"][1] = deepcopy(row)
    elif fault == "wrong_target":
        row["target_quote"] = "This text is not in the target."
    elif fault == "wrong_index":
        row["message_index"] = 1
    elif fault == "contradiction":
        row["reason"] = "This is a bad turn."
    else:
        row.update(verdict="bad", refusal_requests=[{"target": "A", "operation": "cancel"}])
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(answer)}}]})

    judge = mock_client(monkeypatch, scratch_dir, respond)
    with pytest.raises(JudgeServiceFailure):
        asyncio.run(evaluate_scoring_inputs(judge, inputs))
    assert len(calls) == 1
    assert not list((scratch_dir / MODE).glob("*.json"))
    audit = json.loads(next((scratch_dir / MODE / "failures").glob("*.json")).read_text())
    assert audit["attempts"][0]["raw_response"] == json.dumps(answer)
    assert audit["last_error"]


def test_quote_and_message_index_anchor_turn_without_repeated_ids():
    inputs = frozen_inputs()
    answer = response_for(inputs)
    baseline = decisions_from_result(
        TrajectoryOnceResult.model_validate(answer), inputs["simple_turn_inputs"]
    )
    for row in answer["turns"]:
        row.pop("turn_index")
    answer["turns"][0]["evidence_message_indices"] = []
    answer["turns"][1]["evidence_message_indices"] = [3, 3]
    answer["turns"].reverse()
    decisions = decisions_from_result(
        TrajectoryOnceResult.model_validate(answer), inputs["simple_turn_inputs"]
    )
    assert decisions[0] == baseline[0]
    assert decisions[1] == {**baseline[1], "evidence_message_indices": [3, 4]}
    assert answer["turns"][0]["evidence_message_indices"] == [3, 3]


def test_one_frozen_retry_lists_both_invalid_target_rows(monkeypatch, scratch_dir):
    inputs = frozen_inputs()
    good = response_for(inputs)
    bad = deepcopy(good)
    bad["turns"][0]["target_quote"] = "Different target quote from the user."
    bad["turns"][1]["evidence_message_indices"].append(5)
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        result = bad if len(calls) == 1 else good
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)}}]})

    judge = mock_client(monkeypatch, scratch_dir, respond, max_retries=1)
    _, raw, _, _ = asyncio.run(evaluate_scoring_inputs(judge, inputs))
    assert json.loads(raw) == good
    assert len(calls) == 2
    assert calls[1]["messages"][:-1] == calls[0]["messages"]
    feedback = calls[1]["messages"][-1]["content"]
    assert "message_index 2 (cite 0..2)" in feedback
    assert "message_index 4 (cite 0..4)" in feedback
    asyncio.run(evaluate_scoring_inputs(judge, inputs))
    assert len(calls) == 2


def test_missing_bulk_response_never_calls_per_turn(monkeypatch, scratch_dir):
    def forbidden(request):
        pytest.fail("Must not fall back to per-turn HTTP requests")

    judge = mock_client(monkeypatch, scratch_dir, forbidden)
    with pytest.raises(ValueError, match="response"):
        asyncio.run(score_simple_turns(
            frozen_inputs()["simple_turn_inputs"], [], {}, judge, mode=MODE))


def test_formal_profile_enforces_single_request_and_disables_user_filter():
    root = Path(__file__).parents[1]
    project = load_yaml(root / "configs/rl/airline_procredit_v4_train30.yaml")
    validate_procredit_config(project)
    assert project["judge"]["mode"] == MODE
    assert project["user_sim_filter"]["enabled"] is False
    assert project["rollout"]["max_hard_turns"] == 35
    for key, bad_value in [("mode", "unknown"), ("calls_per_trajectory", 2)]:
        invalid = deepcopy(project)
        invalid["judge"][key] = bad_value
        with pytest.raises(ValueError, match="judge"):
            validate_procredit_config(invalid)


def test_combined_validator_retains_global_rubric_checks():
    inputs = frozen_inputs()
    inputs["judge"]["semantic_checks"] = [{"criterion_id": "must-answer"}]
    parsed = TrajectoryOnceJudge.result_model.model_validate(response_for(inputs))
    with pytest.raises(ValueError, match="criterion IDs"):
        TrajectoryOnceJudge._validate_requested_criteria(parsed, {
            **inputs["judge"], "turn_inputs": inputs["simple_turn_inputs"]})


@pytest.mark.parametrize(
    "envelope",
    [{"choices": []}, {"choices": [{"message": {"content": None}}]}, []],
)
def test_bad_http_envelope_retries_without_reprompting(monkeypatch, scratch_dir, envelope):
    inputs = frozen_inputs()
    requests = []

    async def no_sleep(_seconds):
        pass

    monkeypatch.setattr("tau2_agentic_rl.judge.client.asyncio.sleep", no_sleep)

    def respond(request):
        requests.append(request.content)
        if len(requests) == 1:
            return httpx.Response(200, json=envelope)
        return httpx.Response(200, json={"choices": [
            {"message": {"content": json.dumps(response_for(inputs))}}
        ]})

    judge = mock_client(monkeypatch, scratch_dir, respond, max_retries=2)
    heartbeats = []
    asyncio.run(evaluate_scoring_inputs(judge, inputs, progress=lambda: heartbeats.append(1)))
    assert len(requests) == len(heartbeats) == 2
    assert requests[0] == requests[1]


def test_total_request_deadline_retries_and_releases_the_lease(monkeypatch, scratch_dir):
    from tau2_agentic_rl.concurrency import api_budget

    started, completed = [], []

    async def no_sleep(_seconds):
        pass

    async def wait_for_response(_request):
        started.append(1)
        await asyncio.Event().wait()

    monkeypatch.setattr("tau2_agentic_rl.judge.client.asyncio.sleep", no_sleep)
    judge = mock_client(monkeypatch, scratch_dir, wait_for_response, max_retries=2)
    judge.config = replace(judge.config, timeout_seconds=0.02)

    async def run():
        with pytest.raises(JudgeServiceFailure) as caught:
            await asyncio.wait_for(judge.evaluate(
                **frozen_inputs()["judge"], on_request_end=lambda: completed.append(1),
            ), timeout=2)
        assert caught.value.retryable
        assert "total deadline" in str(caught.value)
        assert api_budget().call("snapshot")["active"]["judge_api"] == 0

    asyncio.run(run())
    assert len(started) == len(completed) == 3


@pytest.mark.parametrize("failed_http_attempts", [0, 6])
def test_live_rollout_and_frozen_retry_share_one_request(
    monkeypatch, scratch_dir, failed_http_attempts,
):
    from test_procredit_rollout import rollout_fixture

    from tau2_agentic_rl.scoring_retry import retry_scoring

    calls = []
    original_sleep = asyncio.sleep

    async def quick_sleep(_seconds):
        await original_sleep(0)

    monkeypatch.setattr("tau2_agentic_rl.judge.client.asyncio.sleep", quick_sleep)

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        timeline = json.loads(payload["messages"][1]["content"].split(
            "ACTOR_VISIBLE_TIMELINE_AND_TARGETS:\n")[1])
        turns = []
        for target in timeline["targets"]:
            message = timeline["visible_messages"][target["message_index"]]["message"]
            quote = message.get("raw_generated_text") or message["content"]
            turns.append({**target, "target_quote": quote[:60], "verdict": "neutral",
                          "evidence_message_indices": [target["message_index"]],
                          "reason": "The reply reports the supported refund.", "refusal_requests": []})
        answer = {"trajectory_result": {"semantic_checks": [], "transfer_semantic_checks": [],
                                        "mandatory_policy_checks": [], "transfer_check": {}},
                  "turns": turns}
        if len(calls) <= failed_http_attempts:
            answer["turns"][0]["target_quote"] = "This is not a quotation from that turn."
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(answer)}}]})

    judge = mock_client(monkeypatch, scratch_dir / "cache", respond, max_retries=2)
    loop, output, record = rollout_fixture(
        scratch_dir / "run", credit_version="procredit-turn-v4", final_turn_rule="simple-turn-v2",
        hard_limit=2, natural_stop_at=2, judge_mode=MODE, actual_judge=judge)
    assert output.reward_score == record.custom_reward.train_reward
    assert record.assistant_turns == 2
    assert record.user_sim_inputs is None and record.user_sim_result is None
    assert record.metadata["simple_turn_labels"]["judge_mode"] == MODE
    assert len(calls) == failed_http_attempts + 1
    assert all(call["messages"][:2] == calls[0]["messages"] for call in calls)
    if failed_http_attempts:
        assert [row["success"] for row in record.metadata["scoring_retries"]] == [False, True]
    before = deepcopy(record.model_dump())
    record.custom_reward = None
    record.metadata["failure_phase"] = "turn_labels"
    assert asyncio.run(retry_scoring(record, judge, loop.store))
    assert len(calls) == failed_http_attempts + 1
    for field in ("messages", "environment_transcript", "token_turns", "scoring_inputs", "custom_reward"):
        assert record.model_dump()[field] == before[field]
    assert record.metadata["simple_turn_labels"] == before["metadata"]["simple_turn_labels"]


def test_prompt_has_one_combined_output_schema_and_retains_receipt_rules():
    inputs = frozen_inputs()
    messages = build_messages(**inputs["judge"], turn_inputs=inputs["simple_turn_inputs"])
    rubric_text = messages[1]["content"].split("FIXED_TASK_AND_RUBRIC:\n")[1].split(
        "\n\nTRAJECTORY_DATA_UNTRUSTED:")[0]
    schema = json.loads(rubric_text)["rubric"]["output_schema"]
    assert set(schema) == {"trajectory_result", "turns"}
    assert set(schema["trajectory_result"]) == {
        "schema_version", "semantic_checks", "transfer_semantic_checks", "mandatory_policy_checks", "transfer_check"}
    assert set(schema["turns"][0]) == {
        "message_index", "target_quote", "verdict", "evidence_message_indices", "reason", "refusal_requests"}
    assert "original rubric output_schema" not in messages[0]["content"]
    assert "Its own receipt is always out of scope" in messages[0]["content"]


def test_flattened_smoke_response_is_validated_once_and_reused_offline(monkeypatch, scratch_dir):
    inputs = frozen_inputs()
    expected = response_for(inputs)
    flattened = {"schema_version": "1.0", **expected["trajectory_result"], "turns": expected["turns"]}
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(flattened)}}]})

    judge = mock_client(monkeypatch, scratch_dir, respond)

    async def run():
        result, raw, prompt_hash, cache_key = await evaluate_scoring_inputs(judge, inputs)
        trace = await score_simple_turns(
            inputs["simple_turn_inputs"], [], {"violating_turns": [], "attribution_complete": True},
            judge, mode=MODE, batch_response=(raw, prompt_hash, cache_key))
        original = TrajectoryOnceResult.model_validate(expected).trajectory_result
        DeepSeekJudge._validate_requested_criteria(original, inputs["judge"])
        assert result.model_dump() == original.model_dump()
        assert json.loads(raw) == flattened  # Preserve the exact model response for audit.
        return trace

    first = asyncio.run(run())
    assert asyncio.run(run()) == first
    assert len(calls) == 1


@pytest.mark.parametrize("flattened", [False, True])
def test_json_object_transport_marker_is_ignored_without_mutating_response(flattened):
    inputs = frozen_inputs()
    expected = response_for(inputs)
    body = {**expected["trajectory_result"], "turns": expected["turns"]} if flattened else expected
    raw = {"type": "json_object", **body}
    result = TrajectoryOnceResult.model_validate(raw)
    assert result == TrajectoryOnceResult.model_validate(expected)
    assert raw["type"] == "json_object"
    TrajectoryOnceJudge._validate_requested_criteria(
        result, {**inputs["judge"], "turn_inputs": inputs["simple_turn_inputs"]},
    )


@pytest.mark.parametrize("fault", [
    "extra", "mixed", "missing_turns", "missing_global_check", "future_receipt", "wrong_type",
])
def test_flat_wrapper_compatibility_cannot_relax_verdict_validation(fault):
    inputs = frozen_inputs()
    expected = response_for(inputs)
    flat = {**expected["trajectory_result"], "turns": expected["turns"]}
    if fault == "wrong_type":
        flat["type"] = "arbitrary"
    elif fault == "extra":
        flat["type"] = "json_object"
        flat["unknown"] = True
    elif fault == "mixed":
        flat["trajectory_result"] = expected["trajectory_result"]
    elif fault == "missing_turns":
        flat.pop("turns")
    elif fault == "missing_global_check":
        inputs["judge"]["semantic_checks"] = [{"criterion_id": "must-answer"}]
    else:
        flat["turns"][0]["evidence_message_indices"].append(3)
    with pytest.raises(ValueError):
        parsed = TrajectoryOnceResult.model_validate(flat)
        TrajectoryOnceJudge._validate_requested_criteria(
            parsed, {**inputs["judge"], "turn_inputs": inputs["simple_turn_inputs"]})

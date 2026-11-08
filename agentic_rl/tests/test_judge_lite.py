import json

import httpx
import pytest
from test_simple_turn_reward import decision, inputs

from tau2_agentic_rl.failures import JudgeServiceFailure
from tau2_agentic_rl.judge.client import DeepSeekJudge, JudgeConfig
from tau2_agentic_rl.judge.simple_turn import SimpleTurnJudge, score_simple_turns
from tau2_agentic_rl.reward.simple_turn import TurnDecision, turn_request
from tau2_agentic_rl.versions import sha256_json


@pytest.mark.parametrize(
    "reason",
    [
        "This is a bad turn.",
        "I'll mark bad_turn true.",
        "The quoted total is wrong, making it a bad turn.",
    ],
)
def test_affirmative_error_cannot_be_neutral(reason):
    request = turn_request(inputs(1), 0, [])
    value = decision(0, request=request)
    value["reason"] = reason
    with pytest.raises(ValueError, match="contradiction"):
        SimpleTurnJudge._validate_requested_criteria(
            TurnDecision.model_validate(value), request
        )
    value["bad_turn"] = True
    SimpleTurnJudge._validate_requested_criteria(
        TurnDecision.model_validate(value), request
    )


@pytest.mark.parametrize(
    "reason",
    [
        "I cannot conclude that this is a bad turn.",
        "If this is a bad turn, more evidence is needed.",
        "The evidence does not support bad_turn=true.",
        "If the quote were wrong, making it a bad turn.",
        "This is not a bad turn.",
        "I will not mark bad_turn true.",
        'The user quoted "This is a bad turn."',
        "It would be incorrect to say this is a bad turn.",
        "The user did not authorize payment, making it a bad turn.",
    ],
)
def test_negation_quotes_and_actual_violation_have_distinct_meanings(reason):
    request = turn_request(inputs(1), 0, [])
    value = decision(0, request=request)
    value["reason"] = reason
    if reason.startswith("The user did not authorize"):
        with pytest.raises(ValueError, match="contradiction"):
            SimpleTurnJudge._validate_requested_criteria(
                TurnDecision.model_validate(value), request
            )
    else:
        SimpleTurnJudge._validate_requested_criteria(
            TurnDecision.model_validate(value), request
        )


def test_current_error_cannot_also_claim_a_clean_refusal():
    request = turn_request(inputs(1), 0, [])
    value = decision(
        0,
        request=request,
        bad=True,
        refusal=True,
        target="A",
        operation="remove_passenger",
    )
    with pytest.raises(ValueError, match="contradiction"):
        SimpleTurnJudge._validate_requested_criteria(
            TurnDecision.model_validate(value), request
        )


@pytest.mark.asyncio
async def test_consistency_feedback_uses_existing_retry_and_preserves_wire_audit(
    monkeypatch, scratch_dir
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-fixture-key")
    request = turn_request(inputs(1), 0, [])
    bodies = []

    def respond(wire):
        body = json.loads(wire.content)
        bodies.append(body)
        value = decision(0, request=request, bad=len(bodies) > 1)
        value["reason"] = "This is a bad turn."
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(value)}}]}
        )

    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(respond), **kw),
    )
    client = SimpleTurnJudge(
        JudgeConfig(model="fixture", cache_dir=str(scratch_dir), max_retries=1)
    )
    result, _, _, key = await client.evaluate(**request)
    assert result.bad_turn and len(bodies) == 2
    assert bodies[1]["messages"][:2] == bodies[0]["messages"]
    assert "contradiction" in bodies[1]["messages"][-1]["content"]
    envelope = json.loads((scratch_dir / (key + ".json")).read_text())
    assert envelope["accepted_request_sha256"] == sha256_json(bodies[1]["messages"])
    assert len(envelope["validation_history"]) == 1
    assert (await client.evaluate(**request))[0] == result and len(bodies) == 2


@pytest.mark.asyncio
async def test_persistent_contradiction_stays_unresolved_with_bounded_attempts(
    monkeypatch, scratch_dir
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-fixture-key")
    request = turn_request(inputs(1), 0, [])
    calls = []

    def respond(wire):
        calls.append(True)
        value = decision(0, request=request)
        value["reason"] = "This is a bad turn."
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(value)}}]}
        )

    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(respond), **kw),
    )
    client = SimpleTurnJudge(
        JudgeConfig(model="fixture", cache_dir=str(scratch_dir), max_retries=1)
    )
    with pytest.raises(JudgeServiceFailure):
        await client.evaluate(**request)
    assert len(calls) == 2 and not list(scratch_dir.glob("*.json"))


@pytest.mark.asyncio
async def test_pipeline_has_only_one_call_per_turn_and_unchanged_flat_schema(
    monkeypatch, scratch_dir
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-fixture-key")
    calls = []

    def respond(wire):
        body = json.loads(wire.content)
        request = json.loads(body["messages"][1]["content"])
        calls.append(request)
        value = decision(request["turn_index"], request=request)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(value)}}]}
        )

    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(respond), **kw),
    )
    client = DeepSeekJudge(JudgeConfig(model="fixture", cache_dir=str(scratch_dir)))
    frozen = inputs(3)
    args = (frozen, [], {"violating_turns": [], "attribution_complete": True}, client)
    trace = await score_simple_turns(*args)
    assert len(calls) == 3 and len(trace["judge_calls"]) == 3
    assert set(trace["decisions"][0]) == set(TurnDecision.model_fields)
    assert trace["judge_version"] == "simple-turn-target-v2"
    assert await score_simple_turns(*args) == trace and len(calls) == 3

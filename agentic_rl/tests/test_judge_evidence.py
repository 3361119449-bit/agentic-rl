import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from tau2_agentic_rl.judge import client as judge_client
from tau2_agentic_rl.judge.client import DeepSeekJudge, JudgeConfig
from tau2_agentic_rl.judge.evidence import validate_evidence_turn_ids
from tau2_agentic_rl.judge.prompts import build_judge_messages
from tau2_agentic_rl.schemas import JudgeResult
from tau2_agentic_rl.versions import sha256_json

GROUPS = [
    "semantic_checks",
    "transfer_semantic_checks",
    "mandatory_policy_checks",
    "transfer_check",
]


def judge_inputs():
    return {
        "task": {},
        "policy": "policy",
        "trajectory": {
            "messages": [
                {"role": "user", "content": "hello", "turn_idx": 0},
                {"role": "assistant", "content": "delivered", "turn_idx": 2},
            ],
            "tool_events": [
                {"name": "invalid_attempt", "turn_id": 7, "success": False},
                {"name": "transfer_to_human_agents", "turn_id": 11, "success": True},
            ],
        },
        "semantic_checks": [{"criterion_id": "s"}],
        "mandatory_policy_checks": [{"criterion_id": "p"}],
        "transfer_rule": {"semantic_checks": [{"criterion_id": "t"}]},
    }


def judge_payload():
    return {
        "semantic_checks": [
            {"criterion_id": "s", "passed": True, "evidence_turn_ids": [2]}
        ],
        "transfer_semantic_checks": [
            {"criterion_id": "t", "passed": True, "evidence_turn_ids": [11]}
        ],
        "mandatory_policy_checks": [
            {
                "criterion_id": "p",
                "passed": False,
                "evidence_turn_ids": [7],
                "short_reason": "rejected attempt",
            }
        ],
        "transfer_check": {
            "applicable": True,
            "valid": True,
            "evidence_turn_ids": [11],
        },
    }


def change_evidence(payload, group, ids):
    check = payload[group] if group == "transfer_check" else payload[group][0]
    check["evidence_turn_ids"] = ids


@pytest.mark.parametrize("group", GROUPS)
def test_all_judge_sections_reject_absent_ids_including_array_positions(group):
    inputs, payload = judge_inputs(), judge_payload()
    DeepSeekJudge._validate_requested_criteria(
        JudgeResult.model_validate(payload), inputs
    )
    change_evidence(payload, group, [1])  # Not an ID, despite messages[1] existing.
    with pytest.raises(ValueError, match="absent turn IDs: \\[1\\]"):
        DeepSeekJudge._validate_requested_criteria(
            JudgeResult.model_validate(payload), inputs
        )


@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("invalid", [True, False, -1, 2.0, "2", None])
def test_evidence_ids_are_strict_nonnegative_integers(group, invalid):
    payload = judge_payload()
    change_evidence(payload, group, [invalid])
    with pytest.raises(ValidationError):
        JudgeResult.model_validate(payload)


def test_zero_message_id_and_rejected_tool_event_are_valid_references():
    inputs, payload = judge_inputs(), judge_payload()
    for group in GROUPS:
        change_evidence(payload, group, [0, 2, 7, 11])
    DeepSeekJudge._validate_requested_criteria(
        JudgeResult.model_validate(payload), inputs
    )


def test_empty_evidence_is_allowed_except_existing_failed_policy_requirement():
    payload = judge_payload()
    for group in GROUPS:
        change_evidence(payload, group, [])
    result = JudgeResult.model_validate(payload)
    validate_evidence_turn_ids(result, judge_inputs()["trajectory"])
    with pytest.raises(ValueError, match="requires evidence"):
        DeepSeekJudge._validate_requested_criteria(result, judge_inputs())
    result.mandatory_policy_checks[0].passed = True
    DeepSeekJudge._validate_requested_criteria(result, judge_inputs())


def test_missing_or_coerced_transcript_ids_do_not_create_evidence():
    payload = judge_payload()
    for group in GROUPS:
        change_evidence(payload, group, [0])
    result = JudgeResult.model_validate(payload)
    for messages in [
        [{"content": "missing"}],
        [{"turn_idx": "0"}],
        [{"turn_idx": False}],
    ]:
        with pytest.raises(ValueError, match="absent turn IDs"):
            validate_evidence_turn_ids(result, {"messages": messages})


def install_mock_api(monkeypatch, responses):
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        payload = responses[min(len(calls) - 1, len(responses) - 1)]
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(payload)}}]}
        )

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        judge_client.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(
            **kwargs,
            transport=httpx.MockTransport(handle),
        ),
    )

    # No wall-clock backoff or real API calls in these sandbox tests.
    async def no_sleep(_):
        pass

    monkeypatch.setattr(judge_client.asyncio, "sleep", no_sleep)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-fixture")
    monkeypatch.delenv("AGENTIC_RL_CONFIG", raising=False)
    return calls


def test_invalid_api_evidence_retries_then_caches_only_valid_result(
    scratch_dir, monkeypatch
):
    good, bad = judge_payload(), judge_payload()
    change_evidence(bad, "semantic_checks", [99])
    calls = install_mock_api(monkeypatch, [bad, good])
    judge = DeepSeekJudge(
        JudgeConfig(model="fixture", cache_dir=str(scratch_dir), max_retries=1)
    )
    inputs = judge_inputs()
    result = asyncio.run(judge.evaluate(**inputs))
    assert len(calls) == 2
    assert result[0] == JudgeResult.model_validate(good)
    assert len(list(scratch_dir.glob("*.json"))) == 1
    assert asyncio.run(judge.evaluate(**inputs)) == result
    assert len(calls) == 2  # Valid cache hit is revalidated without API access.


def test_bad_api_evidence_exhausts_bounded_retries_without_saving_score(
    scratch_dir, monkeypatch
):
    bad = judge_payload()
    change_evidence(bad, "transfer_check", [99])
    calls = install_mock_api(monkeypatch, [bad])
    judge = DeepSeekJudge(
        JudgeConfig(model="fixture", cache_dir=str(scratch_dir), max_retries=1)
    )
    with pytest.raises(RuntimeError, match="bounded retries") as error:
        asyncio.run(judge.evaluate(**judge_inputs()))
    assert "absent turn IDs" in str(error.value.__cause__)
    assert len(calls) == 2
    assert not list(scratch_dir.glob("*.json"))


def test_invalid_cached_evidence_is_not_trusted_or_silently_repaired(
    scratch_dir, monkeypatch
):
    judge = DeepSeekJudge(JudgeConfig(model="fixture", cache_dir=str(scratch_dir)))
    inputs, bad = judge_inputs(), judge_payload()
    change_evidence(bad, "mandatory_policy_checks", [99])
    identity = judge.cache_identity(build_judge_messages(**inputs))
    path = scratch_dir / f"{sha256_json(identity)}.json"
    path.write_text(json.dumps({"identity": identity, "result": bad}), encoding="utf-8")
    before = path.read_bytes()
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="absent turn IDs"):
        asyncio.run(judge.evaluate(**inputs))
    assert path.read_bytes() == before

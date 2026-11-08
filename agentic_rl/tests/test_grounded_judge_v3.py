import json

import pytest

from tau2_agentic_rl.reward.simple_turn import build_turn_inputs, turn_request


def fixture_request(text="Your refund is $716."):
    from tau2_agentic_rl.reward.grounded import GROUNDED_VERSION

    inputs = build_turn_inputs(
        [
            {"role": "system", "content": "Bound policy"},
            {
                "role": "user",
                "content": "Please refund my original card; maximum new charge $352.",
            },
            {
                "role": "tool",
                "content": '{"refund":2571,"passengers":2,"fares":[430,412],"old_total":490}',
            },
            {
                "role": "assistant",
                "turn_idx": 1,
                "content": text,
                "raw_generated_text": text,
            },
        ],
        "Bound policy",
        1,
        judge_version=GROUNDED_VERSION,
    )
    return turn_request(inputs, 0, [])


def payload(request, **changes):
    return {
        "turn_index": 0,
        "target_message_index": request["target_message_index"],
        "target_generation_sha256": request["target_generation_sha256"],
        "target_quote": request["target_generation_text"],
        "audited_categories": [
            "amount",
            "authorization",
            "parameters",
            "grounding",
            "policy",
            "transfer_summary",
        ],
        "checks": [],
        "amount_checks": [],
        "refusal_verified": False,
        "refusal_requests": [],
        "revoked_refusals": [],
        "policy_reviews": [],
        "neutral_reason": "No other issue found in the current generation.",
        **changes,
    }


def amount(request, **changes):
    return {
        "kind": "arithmetic",
        "expression": "2787-216",
        "expected": 2571,
        "reported": 716,
        "authorized": None,
        "candidates": None,
        "rank": None,
        "quote": request["target_generation_text"],
        "evidence_message_indices": [2, 3],
        "reason": "The actual refund differs from the reported refund.",
        **changes,
    }


def test_amount_mismatch_derives_bad_without_separate_boolean():
    from tau2_agentic_rl.reward.grounded import validate_assessment

    r = fixture_request()
    audit, decision = validate_assessment(payload(r, amount_checks=[amount(r)]), r)
    assert decision["bad_turn"] is True
    assert decision["correct_refusal"] is False
    assert audit["amount_checks"][0]["expected"] == 2571


def test_inconsistent_model_arithmetic_is_rejected_for_retry():
    from tau2_agentic_rl.reward.grounded import validate_assessment

    r = fixture_request()
    with pytest.raises(ValueError, match="arithmetic"):
        validate_assessment(payload(r, amount_checks=[amount(r, expected=716)]), r)


def test_two_passenger_charge_exceeding_authorization_is_bad():
    from tau2_agentic_rl.reward.grounded import validate_assessment

    r = fixture_request('update_reservation_flights({"cabin":"business"})')
    check = amount(
        r,
        kind="authorization",
        expression="(430+412)*2-490",
        expected=1194,
        reported=None,
        authorized=352,
    )
    _, decision = validate_assessment(payload(r, amount_checks=[check]), r)
    assert decision["bad_turn"] is True


def test_rank_audit_includes_direct_option_and_rejects_wrong_expected_rank():
    from tau2_agentic_rl.reward.grounded import validate_assessment

    r = fixture_request("The second cheapest option costs $331.")
    audit = amount(
        r,
        kind="ranking",
        expression=None,
        candidates=[163, 300, 290, 304, 331],
        rank=2,
        expected=290,
        reported=331,
    )
    _, d = validate_assessment(payload(r, amount_checks=[audit]), r)
    assert d["bad_turn"] is True
    with pytest.raises(ValueError, match="ranking"):
        validate_assessment(payload(r, amount_checks=[{**audit, "expected": 300}]), r)


def test_error_wins_over_refusal_and_unknown_cannot_get_clean_refusal():
    from tau2_agentic_rl.reward.grounded import validate_assessment

    r = fixture_request(
        "The airline has not cancelled your flight; I cannot refund it."
    )
    finding = {
        "category": "grounding",
        "verdict": "error",
        "quote": r["target_generation_text"],
        "reason": "No flight status supports this definite claim.",
        "evidence_message_indices": [3],
    }
    refusals = [{"target": "booking", "operation": "cancel"}]
    _, d = validate_assessment(
        payload(r, checks=[finding], refusal_verified=True, refusal_requests=refusals),
        r,
    )
    assert (
        d["bad_turn"] is True
        and d["correct_refusal"] is False
        and not d["refusal_requests"]
    )
    with pytest.raises(ValueError, match="uncertain"):
        validate_assessment(
            payload(
                r,
                checks=[{**finding, "verdict": "uncertain"}],
                refusal_verified=True,
                refusal_requests=refusals,
            ),
            r,
        )


def test_future_evidence_and_unbound_quotes_never_pass():
    from tau2_agentic_rl.reward.grounded import validate_assessment

    r = fixture_request()
    for check in [
        amount(r, evidence_message_indices=[2, 4]),
        amount(r, quote="not in target"),
    ]:
        with pytest.raises(ValueError):
            validate_assessment(payload(r, amount_checks=[check]), r)


@pytest.mark.parametrize(
    "reason",
    [
        "This is a bad turn.",
        "I'll mark bad_turn true.",
        "The ranking is wrong, making it a bad turn.",
    ],
)
def test_affirmative_current_error_reason_cannot_be_neutral(reason):
    from tau2_agentic_rl.reward.grounded import validate_assessment

    r = fixture_request()
    with pytest.raises(ValueError, match="contradiction"):
        validate_assessment(payload(r, neutral_reason=reason), r)


@pytest.mark.parametrize(
    "text",
    ["I found your reservation.", "The certificate would cover $299, leaving $201."],
)
def test_ambiguous_receipt_and_hypothetical_balance_can_remain_neutral(text):
    from tau2_agentic_rl.reward.grounded import validate_assessment

    r = fixture_request(text)
    _, d = validate_assessment(payload(r), r)
    assert d["bad_turn"] is False and d["correct_refusal"] is False


def test_legacy_v2_target_contract_is_unchanged_and_versions_do_not_mix():
    from tau2_agentic_rl.reward.grounded import GROUNDED_VERSION

    old = build_turn_inputs(
        [{"role": "assistant", "turn_idx": 1, "content": "Hello"}], "Policy", 1
    )
    assert turn_request(old, 0, [])["request_version"] == "simple-turn-target-v2"
    new = build_turn_inputs(
        old["messages"], "Policy", 1, judge_version=GROUNDED_VERSION
    )
    assert (
        turn_request(new, 0, [])["target_generation_sha256"]
        != turn_request(old, 0, [])["target_generation_sha256"]
    )


@pytest.mark.asyncio
async def test_live_client_retries_inconsistent_response_with_scoped_feedback(
    monkeypatch, scratch_dir
):
    import httpx

    from tau2_agentic_rl.judge.client import JudgeConfig
    from tau2_agentic_rl.judge.grounded import GroundedTurnJudge
    from tau2_agentic_rl.reward.grounded import validate_assessment

    r = fixture_request()
    bad = payload(r, neutral_reason="This is a bad turn.")
    good = payload(r, amount_checks=[amount(r)])
    bodies = []

    def responder(request):
        body = json.loads(request.content)
        bodies.append(body)
        answer = bad if len(bodies) == 1 else good
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(answer)}}]}
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(responder), **kw),
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-key")
    judge = GroundedTurnJudge(
        JudgeConfig(
            model="fixture", cache_dir=str(scratch_dir / "cache"), max_retries=1
        )
    )
    result, _, _, key = await judge.evaluate(**r)
    _, decision = validate_assessment(result.model_dump(), r)
    assert decision["bad_turn"] is True
    assert len(bodies) == 2
    assert bodies[1]["messages"][:2] == bodies[0]["messages"]
    assert "contradiction" in bodies[1]["messages"][-1]["content"]
    envelope = json.loads((scratch_dir / "cache" / f"{key}.json").read_text())
    assert len(envelope["validation_history"]) == 1
    assert envelope["accepted_request_sha256"]
    cached = await judge.evaluate(**r)
    assert cached[0] == result and len(bodies) == 2

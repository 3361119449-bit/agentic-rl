from copy import deepcopy

import pytest

from tau2_agentic_rl.schemas import JudgeResult

POLICY = "Basic economy flights cannot be modified. All reservations can change cabin without changing flights. Obtain explicit user confirmation before a write."


def inputs():
    return {
        "judge_validation_version": "judge-grounded-v3",
        "policy": POLICY,
        "trajectory": {
            "messages": [
                {"role": "user", "turn_idx": 0, "content": "Please review my booking."},
                {
                    "role": "assistant",
                    "turn_idx": 1,
                    "assistant_turn_id": 1,
                    "content": "The incorrect old card is 5018.",
                },
                {
                    "role": "user",
                    "turn_idx": 2,
                    "content": "Please charge $163 to my card ending 7334.",
                },
                {
                    "role": "assistant",
                    "turn_idx": 3,
                    "assistant_turn_id": 2,
                    "content": "Charging $163 to card ending 7334.",
                },
            ],
            "tool_events": [],
        },
        "semantic_checks": [],
        "mandatory_policy_checks": [{"criterion_id": "p:confirmation"}],
        "transfer_rule": {},
    }


def raw_result():
    return {
        "schema_version": "1.0",
        "grounding_version": "judge-grounded-v3",
        "semantic_checks": [],
        "transfer_semantic_checks": [],
        "transfer_check": {"applicable": False, "valid": False},
        "mandatory_policy_checks": [
            {
                "criterion_id": "p:confirmation",
                "applicable": True,
                "passed": False,
                "evidence_turn_ids": [1, 3],
                "violation_assistant_turn_ids": [1, 2],
                "short_reason": "Both generations used wrong cards.",
            }
        ],
        "policy_attributions": [
            {
                "criterion_id": "p:confirmation",
                "assistant_turn_id": 1,
                "quote": "The incorrect old card is 5018.",
                "policy_quote": "Obtain explicit user confirmation before a write.",
                "evidence_message_indices": [0, 1],
                "reason": "Old error.",
            },
            {
                "criterion_id": "p:confirmation",
                "assistant_turn_id": 2,
                "quote": "Charging $163 to card ending 7334.",
                "policy_quote": "Obtain explicit user confirmation before a write.",
                "evidence_message_indices": [2, 3],
                "reason": "Alleged unconfirmed current write.",
            },
        ],
    }


def test_grounded_global_failure_requires_one_exact_current_witness_per_blame():
    from tau2_agentic_rl.judge.grounded_policy import validate_global_witnesses

    result = JudgeResult.model_validate(raw_result())
    validate_global_witnesses(result, inputs())
    result.policy_attributions.pop()
    with pytest.raises(ValueError, match="witness"):
        validate_global_witnesses(result, inputs())


@pytest.mark.parametrize(
    "field,value",
    [
        ("quote", "not in current generation"),
        ("policy_quote", "imaginary policy"),
        ("evidence_message_indices", [1, 2]),
        ("evidence_message_indices", [1, 4]),
    ],
)
def test_bad_quote_policy_or_causal_witness_cannot_be_accepted(field, value):
    from tau2_agentic_rl.judge.grounded_policy import validate_global_witnesses

    data = raw_result()
    data["policy_attributions"][0][field] = value
    with pytest.raises(ValueError):
        validate_global_witnesses(JudgeResult.model_validate(data), inputs())


def test_rejected_current_blame_does_not_erase_independent_old_error():
    from tau2_agentic_rl.judge.grounded_policy import reconcile_witnesses

    original = JudgeResult.model_validate(raw_result())
    before = deepcopy(original.model_dump())
    review = {
        "criterion_id": "p:confirmation",
        "verdict": "rejected",
        "quote": "Charging $163 to card ending 7334.",
        "policy_quote": "Obtain explicit user confirmation before a write.",
        "evidence_message_indices": [2, 3],
        "reason": "The latest user explicitly authorized this exact card and amount.",
    }
    revised = reconcile_witnesses(original, 2, [review], inputs())
    assert original.model_dump() == before
    assert revised.mandatory_policy_checks[0].violation_assistant_turn_ids == [1]
    assert revised.mandatory_policy_checks[0].passed is False
    assert len(revised.policy_attributions) == 1
    assert revised.transfer_check == original.transfer_check


def test_rejection_removes_false_policy_failure_but_not_unknown_review():
    from tau2_agentic_rl.judge.grounded_policy import reconcile_witnesses

    d = raw_result()
    d["policy_attributions"] = d["policy_attributions"][1:]
    d["mandatory_policy_checks"][0]["violation_assistant_turn_ids"] = [2]
    result = JudgeResult.model_validate(d)
    review = {
        "criterion_id": "p:confirmation",
        "verdict": "rejected",
        "quote": "Charging $163 to card ending 7334.",
        "policy_quote": "Obtain explicit user confirmation before a write.",
        "evidence_message_indices": [2, 3],
        "reason": "Latest authorization matches current action.",
    }
    revised = reconcile_witnesses(result, 2, [review], inputs())
    assert revised.mandatory_policy_checks[0].passed is True
    assert not revised.mandatory_policy_checks[0].violation_assistant_turn_ids
    with pytest.raises(ValueError, match="unresolved"):
        reconcile_witnesses(result, 2, [{**review, "verdict": "uncertain"}], inputs())


def test_old_global_output_serialization_stays_identical():
    raw = {
        "schema_version": "1.0",
        "semantic_checks": [],
        "transfer_semantic_checks": [],
        "mandatory_policy_checks": [],
        "transfer_check": {
            "applicable": False,
            "valid": False,
            "evidence_turn_ids": [],
            "short_reason": "",
        },
    }
    assert JudgeResult.model_validate(raw).model_dump() == raw


def _turn_inputs():
    from tau2_agentic_rl.reward.simple_turn import GROUNDED_VERSION, build_turn_inputs

    messages = deepcopy(inputs()["trajectory"]["messages"])
    messages[1]["turn_idx"] = 1
    messages[3]["turn_idx"] = 2
    return build_turn_inputs(messages, POLICY, 2, judge_version=GROUNDED_VERSION)


def _assessment(request, *, bad=False, review=False):
    from tau2_agentic_rl.reward.grounded import CATEGORIES

    quote = request["target_generation_text"]
    return {
        "turn_index": request["turn_index"],
        "target_message_index": request["target_message_index"],
        "target_generation_sha256": request["target_generation_sha256"],
        "target_quote": quote,
        "audited_categories": list(CATEGORIES),
        "checks": [
            {
                "category": "grounding",
                "verdict": "error",
                "quote": quote,
                "evidence_message_indices": [request["target_message_index"]],
                "reason": "Visible old card error.",
            }
        ]
        if bad
        else [],
        "amount_checks": [],
        "refusal_verified": False,
        "refusal_requests": [],
        "revoked_refusals": [],
        "policy_reviews": [
            {
                "criterion_id": x["criterion_id"],
                "verdict": "rejected",
                "quote": quote,
                "policy_quote": "Obtain explicit user confirmation before a write.",
                "evidence_message_indices": [2, 3],
                "reason": "Latest user authorized precisely this amount and card.",
            }
            for x in request.get("policy_candidates", [])
        ],
        "neutral_reason": "Current action matches visible authorization.",
    }


@pytest.mark.asyncio
async def test_conflict_review_only_for_neutral_current_turn_and_replays(monkeypatch):
    from tau2_agentic_rl.judge import grounded
    from tau2_agentic_rl.reward.grounded import GroundedAssessment

    calls = []

    class Client:
        async def evaluate(self, **request):
            calls.append(request)
            wire = _assessment(request, bad=request["turn_index"] == 0)
            return GroundedAssessment.model_validate(wire), "raw", "prompt", "cache"

    monkeypatch.setattr(grounded, "client_from", lambda judge: Client())
    original = JudgeResult.model_validate(raw_result())
    result, prepared = await grounded.prepare_grounded_turns(
        _turn_inputs(), inputs(), original, object()
    )
    assert len(calls) == 3
    assert "policy_candidates" not in calls[0] and "policy_candidates" not in calls[1]
    assert calls[2]["policy_candidates"][0]["criterion_id"] == "p:confirmation"
    assert "assistant_turn_id" not in calls[2]["policy_candidates"][0]
    assert result.mandatory_policy_checks[0].violation_assistant_turn_ids == [1]
    assert (
        prepared["decisions"][0]["bad_turn"]
        and not prepared["decisions"][1]["bad_turn"]
    )
    assert (
        grounded.validate_prepared(_turn_inputs(), inputs(), result, prepared)
        == prepared["decisions"]
    )
    corrupted = deepcopy(prepared)
    corrupted["assessments"][1]["review"]["policy_reviews"][0]["verdict"] = "confirmed"
    with pytest.raises(ValueError):
        grounded.validate_prepared(_turn_inputs(), inputs(), result, corrupted)


@pytest.mark.asyncio
async def test_confirmed_global_review_becomes_bad_and_preserves_all_witnesses(
    monkeypatch,
):
    from tau2_agentic_rl.judge import grounded
    from tau2_agentic_rl.reward.grounded import GroundedAssessment

    class Client:
        async def evaluate(self, **request):
            wire = _assessment(request, bad=request["turn_index"] == 0)
            if request.get("policy_candidates"):
                wire["policy_reviews"][0]["verdict"] = "confirmed"
                wire["policy_reviews"][0]["reason"] = (
                    "Current write conflicts with confirmation."
                )
            return GroundedAssessment.model_validate(wire), "raw", "prompt", "cache"

    monkeypatch.setattr(grounded, "client_from", lambda judge: Client())
    original = JudgeResult.model_validate(raw_result())
    result, prepared = await grounded.prepare_grounded_turns(
        _turn_inputs(), inputs(), original, object()
    )
    assert prepared["decisions"][1]["bad_turn"]
    assert result == original


def test_live_and_frozen_grounded_paths_validate_same_labels(scratch_dir, monkeypatch):
    import asyncio

    from test_procredit_rollout import rollout_fixture

    from tau2_agentic_rl.judge import grounded
    from tau2_agentic_rl.reward.grounded import GroundedAssessment
    from tau2_agentic_rl.reward.simple_turn import validate_record_labels
    from tau2_agentic_rl.scoring_retry import retry_scoring

    class Client:
        async def evaluate(self, **request):
            return (
                GroundedAssessment.model_validate(_assessment(request)),
                "raw",
                "prompt",
                "cache",
            )

    monkeypatch.setattr(grounded, "client_from", lambda judge: Client())
    loop, _, record = rollout_fixture(
        scratch_dir,
        credit_version="procredit-turn-v4",
        final_turn_rule="simple-turn-v1",
        grounded_judge=True,
        natural_stop_at=1,
    )
    assert (
        record.scoring_inputs["judge"]["judge_validation_version"]
        == "judge-grounded-v3"
    )
    assert record.scoring_inputs["reward_project_config"]["judge"] == {
        "grounding_version": "judge-grounded-v3"
    }
    assert validate_record_labels(record)["judge_version"] == "simple-turn-target-v3"
    assert record.metadata["judge_result_provenance"]["version"] == "grounded-composite-v3"
    expected = record.custom_reward.model_dump()
    before = deepcopy(record.messages)
    record.custom_reward = None
    record.metadata["failure_phase"] = "judge"
    assert asyncio.run(retry_scoring(record, loop.judge, loop.store))
    assert record.messages == before and record.custom_reward.model_dump() == expected
    assert validate_record_labels(record)
    record.metadata["simple_turn_labels"]["grounded_audit"]["assessments"][0][
        "primary"
    ]["neutral_reason"] = "This is a bad turn."
    with pytest.raises(ValueError):
        validate_record_labels(record)


def test_rejected_unsent_generation_witness_is_bound_to_actual_raw_and_prior_prefix():
    from tau2_agentic_rl.judge.grounded_policy import (
        bind_undelivered,
        validate_global_witnesses,
    )

    view = inputs()
    native = view["trajectory"]["messages"][:3]
    bounded = deepcopy(native) + [
        {
            "role": "assistant",
            "turn_idx": 2,
            "raw_generated_text": "wrong mixed reply <tool_call>bad</tool_call>",
        }
    ]
    view["trajectory"] = bind_undelivered(
        {"messages": native, "tool_events": []}, bounded
    )
    data = raw_result()
    data["policy_attributions"][1].update(
        quote="wrong mixed reply", evidence_message_indices=[2]
    )
    result = JudgeResult.model_validate(data)
    validate_global_witnesses(result, view)
    result.policy_attributions[1].evidence_message_indices = [3]
    with pytest.raises(ValueError):
        validate_global_witnesses(result, view)

import json

import pytest

from tau2_agentic_rl.judge.client import DeepSeekJudge
from tau2_agentic_rl.policy_rules import policy_checks
from tau2_agentic_rl.reward.policy_credit import build_policy_credit
from tau2_agentic_rl.reward.score import _judge_policy_checks
from tau2_agentic_rl.schemas import JudgeCheck, JudgeResult, RewardResult, ToolEvent


def test_inactive_policy_is_not_forced_active_or_blameworthy():
    check = JudgeCheck(criterion_id="x:policy:booking_cabin", applicable=False, passed=True,
                       short_reason="No booking occurred")
    judge = JudgeResult(mandatory_policy_checks=[check])
    assert _judge_policy_checks(judge)[0].applicable is False
    reward = RewardResult(branch="normal", train_reward=0, strict_success=0,
                          progress=0, policy_gate=True, process_penalty=0)
    assert build_policy_credit(reward=reward, judge=judge, events=[], turns=1)["violating_turns"] == []


@pytest.mark.parametrize("changes", [{"passed": False}, {"violation_assistant_turn_ids": [1]}])
def test_na_cannot_carry_failure_or_blame(changes):
    with pytest.raises(ValueError):
        JudgeCheck(criterion_id="p", applicable=False, **({"passed": True} | changes))


def test_new_rubric_requires_explicit_applicability():
    rubric = policy_checks("0")
    result = JudgeResult(mandatory_policy_checks=[
        JudgeCheck(criterion_id=c["criterion_id"], passed=True) for c in rubric])
    with pytest.raises(ValueError, match="applicability"):
        DeepSeekJudge._validate_requested_criteria(result, {"mandatory_policy_checks": rubric})


def test_executed_write_cannot_hide_confirmation_rule_as_na():
    criterion = "0:policy:database_write_confirmation"
    result = JudgeResult(mandatory_policy_checks=[
        JudgeCheck(criterion_id=criterion, applicable=False, passed=True, short_reason="N/A")])
    event = ToolEvent(event_id="e", sequence=0, turn_id=1, name="cancel_reservation", success=True)
    with pytest.raises(ValueError, match="active policy"):
        DeepSeekJudge._validate_requested_criteria(result, {
            "mandatory_policy_checks": [{"criterion_id": criterion}],
            "trajectory": {"messages": [], "tool_events": [event.model_dump()]}})


def test_legacy_fields_do_not_change_frozen_evidence_serialization():
    check = JudgeCheck(criterion_id="p", passed=True)
    assert "applicable" not in check.model_dump()
    event = ToolEvent(event_id="e", sequence=0, turn_id=1)
    assert "state_before" not in event.model_dump()


@pytest.mark.parametrize("rule", [
    "database_write_confirmation", "user_identity", "cancellation_reason", "cancellation_eligibility",
])
def test_failed_real_write_attempt_cannot_hide_preconditions_as_na(rule):
    criterion = f"0:policy:{rule}"
    result = JudgeResult(mandatory_policy_checks=[
        JudgeCheck(criterion_id=criterion, applicable=False, passed=True, short_reason="N/A")])
    event = ToolEvent(event_id="e", sequence=0, turn_id=1, name="cancel_reservation",
                      success=False, db_effect=False, error_kind="model_caused_execution_error")
    with pytest.raises(ValueError, match="active policy"):
        DeepSeekJudge._validate_requested_criteria(result, {
            "mandatory_policy_checks": [{"criterion_id": criterion}],
            "trajectory": {"messages": [], "tool_events": [event.model_dump()]}})


def test_failed_attempt_does_not_invent_completed_refund_or_transfer():
    from tau2_agentic_rl.policy_rules import definitely_active_policy_rules

    events = [
        ToolEvent(event_id=str(i), sequence=i, turn_id=i+1, name=name, success=False,
                  db_effect=False, error_kind="model_caused_execution_error").model_dump()
        for i, name in enumerate(["cancel_reservation", "transfer_to_human_agents"])
    ]
    trajectory = {"messages": [], "tool_events": events}
    assert "refund_destination" not in definitely_active_policy_rules(trajectory, {})
    result = JudgeResult()
    DeepSeekJudge._validate_requested_criteria(result, {"trajectory": trajectory})
    assert not result.transfer_check.applicable and not result.transfer_check.valid


@pytest.mark.parametrize("insurance", ["no", "yes"])
@pytest.mark.parametrize("success", [True, False])
def test_booking_without_insurance_allows_inactive_insurance_check(insurance, success):
    criterion = "12:policy:insurance_timing"
    result = JudgeResult(mandatory_policy_checks=[JudgeCheck(
        criterion_id=criterion, applicable=False, passed=True,
        short_reason="No travel insurance was purchased or added.",
    )])
    event = ToolEvent(
        event_id="booking", sequence=0, turn_id=1, name="book_reservation",
        arguments={"insurance": insurance}, success=success, db_effect=False,
        error_kind=None if success else "model_caused_execution_error",
    )
    inputs = {
        "mandatory_policy_checks": [{"criterion_id": criterion}],
        "trajectory": {"messages": [], "tool_events": [event.model_dump()]},
    }
    if insurance == "yes":
        with pytest.raises(ValueError, match="active policy"):
            DeepSeekJudge._validate_requested_criteria(result, inputs)
    else:
        DeepSeekJudge._validate_requested_criteria(result, inputs)


def test_v4_requires_specific_blame_in_judge_response():
    criterion = "0:policy:information_grounding"
    result = JudgeResult(mandatory_policy_checks=[
        JudgeCheck(criterion_id=criterion, passed=False, applicable=True,
                   evidence_turn_ids=[1], short_reason="invented information")])
    with pytest.raises(ValueError, match="attribution"):
        DeepSeekJudge._validate_requested_criteria(result, {
            "require_policy_attribution": True,
            "mandatory_policy_checks": [{"criterion_id": criterion}],
            "trajectory": {"messages": [
                {"role": "assistant", "turn_idx": 1, "assistant_turn_id": 1, "content": "invented"}
            ]},
        })


def test_v4_missing_required_transfer_cannot_cache_an_unattributable_pass(scratch_dir, monkeypatch):
    import asyncio

    from test_judge_evidence import install_mock_api

    from tau2_agentic_rl.judge.client import JudgeConfig

    criterion = "0:policy:transfer_scope_and_message"
    bad = JudgeResult(mandatory_policy_checks=[JudgeCheck(
        criterion_id=criterion, applicable=True, passed=True,
        short_reason="incorrectly assumes transfer completed",
    )]).model_dump()
    good = JudgeResult(mandatory_policy_checks=[JudgeCheck(
        criterion_id=criterion, applicable=True, passed=False,
        violation_assistant_turn_ids=[1], short_reason="refused required transfer",
    )]).model_dump()
    calls = install_mock_api(monkeypatch, [bad, good])
    judge = DeepSeekJudge(JudgeConfig(model="fixture", cache_dir=str(scratch_dir), max_retries=1))
    inputs = {
        "policy": "transfer required", "task": {}, "semantic_checks": [],
        "mandatory_policy_checks": [{"criterion_id": criterion}],
        "transfer_rule": {"required": True}, "require_policy_attribution": True,
        "trajectory": {"messages": [{
            "role": "assistant", "turn_idx": 1, "assistant_turn_id": 1,
            "content": "I refuse to transfer you.",
        }], "tool_events": []},
    }
    result = asyncio.run(judge.evaluate(**inputs))
    assert len(calls) == 2
    assert result[0].mandatory_policy_checks[0].violation_assistant_turn_ids == [1]
    assert asyncio.run(judge.evaluate(**inputs)) == result
    assert len(calls) == 2
    assert len(list(scratch_dir.glob("*.json"))) == 1


@pytest.mark.parametrize(
    ("tool", "arguments", "after", "rule", "required"),
    [
        ("book_reservation", {"total_baggages": 0, "nonfree_baggages": 0},
         None, "baggage_pricing", False),
        ("book_reservation", {"total_baggages": 1, "nonfree_baggages": 0},
         None, "baggage_pricing", True),
        ("update_reservation_baggages", {"total_baggages": 2, "nonfree_baggages": 0},
         {"total_baggages": 2}, "baggage_removal", False),
        ("update_reservation_baggages", {"total_baggages": 0, "nonfree_baggages": 0},
         None, "baggage_removal", True),
        ("update_reservation_baggages", {"total_baggages": 2, "nonfree_baggages": 0},
         {"total_baggages": 2}, "baggage_pricing", True),
        ("update_reservation_baggages", {"total_baggages": 1, "nonfree_baggages": 1},
         {"nonfree_baggages": 1}, "baggage_pricing", True),
        ("update_reservation_baggages", {"total_baggages": 1, "nonfree_baggages": 0},
         {}, "baggage_pricing", False),
        ("update_reservation_flights",
         {"cabin": "business", "flights": [{"flight_number": "HAT100", "date": "2024-05-20"}]},
         {"cabin": "business"}, "modification_payment", False),
        ("update_reservation_flights",
         {"cabin": "business", "flights": [{"flight_number": "HAT100", "date": "2024-05-21"}]},
         None, "modification_payment", True),
    ],
)
def test_write_applicability_depends_on_arguments_and_prior_receipt(
    tool, arguments, after, rule, required
):
    before = {
        "reservation_id": "old",
        "total_baggages": 1,
        "nonfree_baggages": 0,
        "flights": [{"flight_number": "HAT100", "date": "2024-05-20", "price": 50}],
    }
    read = ToolEvent(
        event_id="read", sequence=0, turn_id=1, name="get_reservation_details",
        arguments={"reservation_id": "old"}, success=True, result=json.dumps(before),
    )
    write = ToolEvent(
        event_id="write", sequence=1, turn_id=2, name=tool,
        arguments={"reservation_id": "old", **arguments}, success=after is not None,
        result=json.dumps({**before, **after}) if after is not None else "Error: rejected",
        error_kind="model_caused_execution_error" if after is None else None,
    )
    criterion = f"14:policy:{rule}"
    result = JudgeResult(mandatory_policy_checks=[JudgeCheck(
        criterion_id=criterion, applicable=False, passed=True, short_reason="No trigger",
    )])
    inputs = {
        "mandatory_policy_checks": [{"criterion_id": criterion}],
        "trajectory": {"messages": [], "tool_events": [read.model_dump(), write.model_dump()]},
    }
    if required:
        with pytest.raises(ValueError, match="active policy"):
            DeepSeekJudge._validate_requested_criteria(result, inputs)
    else:
        DeepSeekJudge._validate_requested_criteria(result, inputs)


@pytest.mark.parametrize(("valid", "evidence", "error"), [
    (False, [4], None),
    (True, [4], "unexecuted transfer"),
    (False, [5], "absent turn IDs"),
])
def test_failed_transfer_can_cite_its_attempt_without_claiming_success(valid, evidence, error):
    criterion = "21:policy:transfer_scope_and_message"
    result = JudgeResult.model_validate({
        "mandatory_policy_checks": [{
            "criterion_id": criterion, "applicable": True, "passed": False,
            "evidence_turn_ids": [4], "violation_assistant_turn_ids": [2],
            "short_reason": "Announced a transfer without executing it.",
        }],
        "transfer_check": {
            "applicable": True, "valid": valid, "policy_required": False,
            "self_caused_only": False, "evidence_turn_ids": evidence,
            "short_reason": "The announcement was not followed by a successful tool call.",
        },
    })
    inputs = {
        "transfer_validation_version": "transfer-policy-v2",
        "require_policy_attribution": True,
        "mandatory_policy_checks": [{"criterion_id": criterion}],
        "trajectory": {"messages": [{
            "role": "assistant", "assistant_turn_id": 2, "turn_idx": 4,
            "content": "I have transferred you.",
        }], "tool_events": []},
    }
    if error:
        with pytest.raises(ValueError, match=error):
            DeepSeekJudge._validate_requested_criteria(result, inputs)
    else:
        DeepSeekJudge._validate_requested_criteria(result, inputs)
        assert not result.transfer_check.valid

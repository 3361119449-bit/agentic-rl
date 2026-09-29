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

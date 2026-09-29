from copy import deepcopy

import pytest

from tau2_agentic_rl.judge.client import DeepSeekJudge
from tau2_agentic_rl.reward.policy_credit import (
    annotate_policy_turns,
    build_policy_credit,
)
from tau2_agentic_rl.schemas import JudgeCheck, JudgeResult, RewardResult, ToolEvent


def reward(**changes):
    values = dict(
        branch="normal",
        reward_mode="strict_progress_v1",
        train_reward=0,
        strict_success=0,
        progress=0,
        policy_gate=False,
        process_penalty=0,
        details={
            "policy_checks": [],
            "task_safety_check": {"rule_id": "safe", "passed": True},
        },
    )
    values.update(changes)
    return RewardResult(**values)


def test_delivered_message_mapping_uses_prefixes_and_excludes_initial_assistant():
    messages = [
        {"role": "assistant", "turn_idx": 0},
        {"role": "user", "turn_idx": 1},
        {"role": "assistant", "turn_idx": 2},
        {"role": "user", "turn_idx": 3},
        {"role": "assistant", "turn_idx": 4},
        {"role": "tool", "turn_idx": 5},
    ]
    original = deepcopy(messages)
    mapped = annotate_policy_turns(messages, [2, 4, 4, 6])
    assert [m.get("assistant_turn_id") for m in mapped] == [
        None,
        None,
        1,
        None,
        3,
        None,
    ]
    assert messages == original


def test_illegal_transfer_localizes_to_executed_tool_without_judge_numeric_guessing():
    event = ToolEvent(
        event_id="transfer",
        sequence=0,
        turn_id=3,
        name="transfer_to_human_agents",
        success=True,
    )
    result = build_policy_credit(
        reward=reward(branch="human_transfer", details={"transfer_valid": False}),
        judge=JudgeResult(
            mandatory_policy_checks=[
                JudgeCheck(
                    criterion_id="4:policy:transfer_scope_and_message",
                    passed=False,
                    evidence_turn_ids=[4, 5],
                    short_reason="out of scope transfer",
                )
            ]
        ),
        events=[event],
        turns=3,
    )
    assert result["violating_turns"] == [2]
    assert result["attribution_complete"]


def test_explicit_violation_ids_do_not_treat_context_evidence_as_offending_turns():
    judge = JudgeResult(
        mandatory_policy_checks=[
            JudgeCheck(
                criterion_id="confirmation",
                passed=False,
                evidence_turn_ids=[0, 3],
                violation_assistant_turn_ids=[2],
                short_reason="write before consent",
            )
        ]
    )
    result = build_policy_credit(reward=reward(), judge=judge, events=[], turns=3)
    assert result["violating_turns"] == [1]
    assert result["attribution_complete"]


def test_old_ambiguous_policy_evidence_is_unresolved_not_guessed():
    judge = JudgeResult(
        mandatory_policy_checks=[
            JudgeCheck(
                criterion_id="grounding",
                passed=False,
                evidence_turn_ids=[2],
                short_reason="bad claim",
            )
        ]
    )
    result = build_policy_credit(reward=reward(), judge=judge, events=[], turns=3)
    assert result["violating_turns"] == []
    assert not result["attribution_complete"]
    assert "grounding" in result["unresolved_checks"]


@pytest.mark.parametrize("blame,complete", [([2], True), ([], False)])
def test_v4_missing_required_transfer_uses_only_explicit_transfer_blame(blame, complete):
    result = build_policy_credit(
        reward=reward(reward_mode="turn_local_v1", details={"policy_checks": [{
            "rule_id": "required_human_transfer_completed", "passed": False,
            "evidence_event_ids": [],
        }]}),
        judge=JudgeResult(mandatory_policy_checks=[JudgeCheck(
            criterion_id="4:policy:transfer_scope_and_message", passed=False,
            evidence_turn_ids=[3], violation_assistant_turn_ids=blame,
            short_reason="actor refused a required transfer",
        )]),
        events=[], turns=3,
    )
    assert result["attribution_complete"] is complete
    assert result["violating_turns"] == ([1] if complete else [])


def test_task_safety_uses_event_ids_and_deduplicates_penalty_turns():
    events = [
        ToolEvent(event_id="write", sequence=0, turn_id=2, name="cancel_reservation")
    ]
    result = build_policy_credit(
        reward=reward(
            task_safety_gate=False,
            details={
                "policy_checks": [
                    {
                        "rule_id": "partial",
                        "passed": False,
                        "evidence_event_ids": ["write"],
                    }
                ],
                "task_safety_check": {
                    "rule_id": "unexpected",
                    "passed": False,
                    "evidence_event_ids": ["write"],
                },
            },
        ),
        judge=JudgeResult(),
        events=events,
        turns=3,
    )
    assert result["violating_turns"] == [1]
    assert result["attribution_complete"]


def test_judge_rejects_violation_ids_that_only_exist_in_other_namespace():
    result = JudgeResult(
        mandatory_policy_checks=[
            JudgeCheck(
                criterion_id="p",
                passed=False,
                evidence_turn_ids=[2],
                short_reason="bad",
                violation_assistant_turn_ids=[2],
            )
        ]
    )
    inputs = {
        "mandatory_policy_checks": [{"criterion_id": "p"}],
        "trajectory": {
            "messages": [{"role": "assistant", "turn_idx": 2, "assistant_turn_id": 1}],
            "tool_events": [],
        },
    }
    with pytest.raises(ValueError, match="assistant"):
        DeepSeekJudge._validate_requested_criteria(result, inputs)

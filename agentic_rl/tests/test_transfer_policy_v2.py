"""Versioned dynamic transfer legality, completion and policy attribution."""
import asyncio

import pytest
from test_judge_evidence import install_mock_api

from tau2_agentic_rl.failures import PolicyAttributionPending
from tau2_agentic_rl.judge.client import DeepSeekJudge, JudgeConfig
from tau2_agentic_rl.judge.prompts import build_judge_messages, rubric_fingerprint
from tau2_agentic_rl.reward.policy_credit import build_policy_credit
from tau2_agentic_rl.reward.score import (
    RewardConfig,
    build_reward_config,
    score_trajectory,
)
from tau2_agentic_rl.schemas import JudgeCheck, JudgeResult, OfficialScores, ToolEvent

VERSION = "transfer-policy-v2"


def event():
    return ToolEvent(event_id="transfer", sequence=1, turn_id=2,
                     name="transfer_to_human_agents", success=True)


def verdict(**changes):
    check = dict(applicable=True, valid=True, policy_required=False,
                 self_caused_only=False, short_reason="The user requested a legal transfer.",
                 evidence_turn_ids=[2])
    check.update(changes)
    return JudgeResult(transfer_check=check)


def score(rule=None, judge=None, events=None, official=None):
    return score_trajectory(
        config=RewardConfig(transfer_validation_version=VERSION),
        events=[event()] if events is None else events,
        messages=[{"role": "assistant", "assistant_turn_id": 1, "content": "Prior response."}],
        assistant_turns=2, required_actions=[], transfer_rule=rule or {},
        judge=verdict() if judge is None else judge,
        official=official or OfficialScores(reward=0),
    )


def test_config_exposes_version_and_preserves_legacy_default():
    assert RewardConfig().transfer_validation_version == "transfer-policy-v1"
    assert build_reward_config({"reward": {"transfer_validation_version": VERSION}}).transfer_validation_version == VERSION
    with pytest.raises(ValueError, match="transfer"):
        RewardConfig(transfer_validation_version="typo")


def test_allowed_false_is_advisory_and_legal_optional_has_no_completion():
    result = score({"allowed": False})
    assert result.details["transfer_valid"] is True
    assert result.policy_gate
    assert result.branch == "normal"
    assert result.strict_success == 0
    assert result.train_reward == 0


@pytest.mark.parametrize("rule,dynamic_required,expected_valid", [
    ({"forbidden": True}, False, False),
    ({"allowed": False, "required": True}, False, True),
    ({"forbidden": True, "required": True}, False, True),
    ({"forbidden": True}, True, True),
])
def test_required_forbidden_precedence(rule, dynamic_required, expected_valid):
    result = score(rule, verdict(policy_required=dynamic_required))
    assert result.details["transfer_valid"] is expected_valid
    assert result.policy_gate is expected_valid
    assert result.strict_success == float(expected_valid and rule.get("required", False))


def test_dynamic_policy_required_transfer_preserves_normal_completion_target():
    result = score({"allowed": False}, verdict(policy_required=True))
    assert result.branch == "normal"
    assert result.strict_success == 0
    assert result.train_reward == 0
    assert result.details["transfer_required"]


def test_legal_optional_transfer_preserves_independently_completed_task():
    result = score({"allowed": False}, official=OfficialScores(reward=1, db_applicable=True, db_score=1))
    assert result.strict_success == 1


def test_self_caused_only_required_transfer_cannot_mint_completion():
    result = score({"required": True}, verdict(policy_required=True, self_caused_only=True))
    assert result.details["transfer_valid"] is True
    assert result.strict_success == 0
    assert result.train_reward == 0


@pytest.mark.parametrize("missing", ["applicable", "valid", "policy_required", "self_caused_only", "short_reason"])
def test_incomplete_v2_verdict_needs_frozen_retry(missing):
    fields = verdict().transfer_check.model_dump()
    fields.pop(missing)
    with pytest.raises(PolicyAttributionPending, match="transfer"):
        score(judge=JudgeResult(transfer_check=fields))


def test_legacy_mode_still_uses_allowed():
    result = score_trajectory(events=[event()], messages=[], assistant_turns=2,
        required_actions=[], official=OfficialScores(reward=0),
        judge=verdict(), transfer_rule={"allowed": False})
    assert result.details["transfer_valid"] is False


def test_dynamic_omission_is_required_and_attribution_remains_unresolved():
    check = verdict(policy_required=True, applicable=True, valid=False, evidence_turn_ids=[])
    result = score(judge=check, events=[])
    assert result.policy_gate is False
    credit = build_policy_credit(reward=result, judge=check, events=[], turns=2)
    assert "required_human_transfer_completed" in credit["unresolved_checks"]


def test_legal_transfer_keeps_unrelated_message_violation_and_prior_bad_turn():
    judge = verdict()
    judge.mandatory_policy_checks = [JudgeCheck.model_validate(item) for item in [
        {"criterion_id": "task:policy:transfer_scope_and_message", "passed": False,
         "violation_assistant_turn_ids": [2], "short_reason": "Transfer message disclosed private data."},
        {"criterion_id": "task:policy:grounding", "passed": False,
         "violation_assistant_turn_ids": [1], "short_reason": "Earlier fabricated result."},
    ]]
    judge = JudgeResult.model_validate(judge.model_dump())
    result = score(judge=judge)
    credit = build_policy_credit(reward=result, judge=judge, events=[event()], turns=2)
    assert credit["violating_turns"] == [0, 1]
    assert credit["attribution_complete"]
    assert not any(row["rule_id"] == "invalid_transfer" for row in credit["evidence"])


def inputs():
    return dict(task={}, policy="Only transfer when user requests it or policy requires.",
        trajectory={"messages": [], "tool_events": [event().model_dump()]},
        semantic_checks=[], mandatory_policy_checks=[], transfer_rule={"allowed": False},
        transfer_validation_version=VERSION)


def test_v2_prompt_and_fingerprint_bind_dynamic_criteria():
    current = inputs()
    messages = build_judge_messages(**current)
    assert "self_caused_only" in messages[1]["content"]
    assert "advisory" in messages[1]["content"]
    assert VERSION in messages[1]["content"]
    old = rubric_fingerprint([], [], {})
    assert old != rubric_fingerprint([], [], {}, transfer_validation_version=VERSION)


def test_client_rejects_missing_new_fields_before_cache():
    with pytest.raises(ValueError, match="transfer"):
        DeepSeekJudge._validate_requested_criteria(JudgeResult(transfer_check={"applicable": True, "valid": True}), inputs())


def test_client_rejects_inconsistent_execution_and_accepts_legal_advisory():
    DeepSeekJudge._validate_requested_criteria(verdict(), inputs())
    with pytest.raises(ValueError, match="transfer"):
        DeepSeekJudge._validate_requested_criteria(verdict(applicable=False), inputs())

def test_required_transfer_missing_explicit_policy_attribution_retries_in_client():
    current = inputs()
    current["trajectory"]["tool_events"] = []
    current["require_policy_attribution"] = True
    judge = verdict(valid=False, policy_required=True, evidence_turn_ids=[])
    with pytest.raises(ValueError, match="attributed policy failure"):
        DeepSeekJudge._validate_requested_criteria(judge, current)


@pytest.mark.parametrize("fields", [
    {"applicable": False},
    {"valid": False, "policy_required": True},
])
def test_no_execution_keeps_explicit_dynamic_decision(fields):
    current = inputs()
    current["trajectory"]["tool_events"] = []
    check = dict(applicable=False, valid=False, policy_required=False,
                 self_caused_only=False, evidence_turn_ids=[], short_reason="No transfer needed.")
    check.update(fields)
    if check["policy_required"]:
        check["applicable"] = True
    judge = JudgeResult(transfer_check=check)
    before = judge.model_dump()
    DeepSeekJudge._validate_requested_criteria(judge, current)
    assert judge.model_dump() == before


def test_invalid_judge_verdict_overrides_required_annotation():
    result = score({"required": True}, verdict(valid=False))
    assert result.strict_success == 0
    assert result.details["transfer_valid"] is False
    assert result.policy_gate is False


def test_v2_does_not_suppress_invalid_transfer_policy_attribution():
    judge = verdict(valid=False)
    result = score(judge=judge)
    credit = build_policy_credit(reward=result, judge=judge, events=[event()], turns=2)
    assert credit["violating_turns"] == [1]
    assert credit["evidence"][0]["rule_id"] == "invalid_transfer"


def test_v2_missing_required_attribution_can_reuse_only_explicit_decision():
    judge = verdict(policy_required=True, valid=False, evidence_turn_ids=[])
    judge.mandatory_policy_checks = [JudgeCheck(
        criterion_id="task:policy:transfer_scope_and_message", passed=False,
        violation_assistant_turn_ids=[1], short_reason="Assistant ended the task without required transfer.")]
    result = score(judge=judge, events=[])
    credit = build_policy_credit(reward=result, judge=judge, events=[], turns=2)
    assert credit["violating_turns"] == [0]
    assert credit["attribution_complete"]

def test_incomplete_remote_verdict_retries_then_caches_only_explicit_result(scratch_dir, monkeypatch):
    bad = {"transfer_check": {"applicable": True, "valid": True}}
    good = verdict().model_dump()
    calls = install_mock_api(monkeypatch, [bad, good])
    client = DeepSeekJudge(JudgeConfig(model="offline-fixture", cache_dir=str(scratch_dir), max_retries=1))
    result = asyncio.run(client.evaluate(**inputs()))
    assert len(calls) == 2
    assert result[0] == verdict()
    assert len(list(scratch_dir.glob("*.json"))) == 1
    assert asyncio.run(client.evaluate(**inputs())) == result
    assert len(calls) == 2


def test_unresolved_remote_verdict_never_caches_or_defaults_valid(scratch_dir, monkeypatch):
    bad = {"transfer_check": {"applicable": True, "valid": True}}
    calls = install_mock_api(monkeypatch, [bad])
    client = DeepSeekJudge(JudgeConfig(model="offline-fixture", cache_dir=str(scratch_dir), max_retries=1))
    with pytest.raises(RuntimeError, match="bounded retries") as error:
        asyncio.run(client.evaluate(**inputs()))
    assert isinstance(error.value.__cause__, PolicyAttributionPending)
    assert len(calls) == 2
    assert not list(scratch_dir.glob("*.json"))


def test_old_new_field_absence_survives_serialization_and_cannot_satisfy_v2():
    old = JudgeResult(transfer_check={"applicable": True, "valid": True, "short_reason": "Old cached verdict."})
    dumped = old.model_dump()
    assert "policy_required" not in dumped["transfer_check"]
    assert "self_caused_only" not in dumped["transfer_check"]
    with pytest.raises(PolicyAttributionPending):
        DeepSeekJudge._validate_requested_criteria(JudgeResult.model_validate(dumped), inputs())


def test_versioned_messages_give_different_judge_cache_keys(scratch_dir):
    from tau2_agentic_rl.versions import sha256_json

    client = DeepSeekJudge(JudgeConfig(model="offline-fixture", cache_dir=str(scratch_dir)))
    old = inputs()
    old.pop("transfer_validation_version")
    old_key = sha256_json(client.cache_identity(build_judge_messages(**old)))
    new_key = sha256_json(client.cache_identity(build_judge_messages(**inputs())))
    assert old_key != new_key

@pytest.mark.parametrize("self_caused,completion", [(True, 0), (False, 1)])
def test_v4_required_transfer_completion_preserves_earlier_bad_turn(self_caused, completion):
    from tau2_agentic_rl.reward.progress import build_progress_trace

    rule = {"required": True, "allowed": False}
    messages = [
        {"role": "assistant", "assistant_turn_id": 1, "content": "A fabricated result."},
        {"role": "assistant", "assistant_turn_id": 2, "content": "Transferring."},
    ]
    judge = verdict(policy_required=True, self_caused_only=self_caused)
    judge.mandatory_policy_checks = [JudgeCheck(
        criterion_id="task:policy:grounding", passed=False,
        violation_assistant_turn_ids=[1], short_reason="Earlier fabricated result.")]
    trace = build_progress_trace(task={}, messages=messages, prefix_lengths=[0, 1, 2],
        events=[event()], required_actions=[], dependencies=[], transfer_rule=rule,
        initial_state_fingerprint="offline-fixture", version="progress-v3")
    config = build_reward_config({
        "credit": {"version": "procredit-turn-v4"},
        "reward": {"mode": "turn_local_v1", "mandatory_policy_gate": False,
                   "task_safety_gate": False, "transfer_validation_version": VERSION},
    })
    result = score_trajectory(config=config, events=[event()], messages=messages,
        assistant_turns=2, required_actions=[], transfer_rule=rule, judge=judge,
        official=OfficialScores(reward=0), progress_trace=trace)
    assert result.details["task_completion"] == completion
    assert result.details["transfer_valid"] is True
    assert result.details["policy_turn_rewards"] == [-1, 0]
    assert result.details["policy_gate_applied"] is False
    assert result.train_reward == completion + 0.5 * trace["phi"][-1]

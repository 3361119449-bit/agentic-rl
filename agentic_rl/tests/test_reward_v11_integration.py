from pathlib import Path

from test_simple_turn_reward import decision, inputs

from tau2_agentic_rl.config import load_yaml
from tau2_agentic_rl.reward.simple_turn import build_label_trace
from tau2_agentic_rl.schemas import ToolEvent


def test_default_reward_v11_and_v10_snapshot_are_separate():
    root = Path(__file__).parents[1]
    current = load_yaml(root / "configs/rl/airline_procredit_v4_v11.yaml")
    assert current["project"]["reward_version"] == "v11-procredit-transfer-query-req"
    assert current["reward"]["progress_version"] == "progress-v3"
    assert current["reward"]["transfer_validation_version"] == "transfer-policy-v2"
    assert current["reward"]["duplicate_query_version"] == "duplicate-query-v1"
    old = load_yaml(root / "configs/rl/airline_procredit_v4_v10.yaml")
    assert old["project"]["reward_version"] == "v10-procredit-simple-turn"


def test_query_evidence_survives_tool_event_serialization():
    event = ToolEvent(
        event_id="q",
        sequence=0,
        turn_id=1,
        name="get_user_details",
        arguments={"user_id": "a"},
        success=True,
        db_effect=False,
        db_hash_before="db",
        db_hash_after="db",
        query_state_before="state",
        query_state_after="state",
        query_state_scope="user:a",
        query_state_version="duplicate-query-v1",
        duplicate_query=True,
        duplicate_of_event_id="earlier",
    )
    saved = event.model_dump()
    assert saved["query_state_before"] == "state"
    assert saved["duplicate_query"] is True


def test_duplicate_uses_existing_bad_turn_guard_even_with_correct_refusal():
    event = ToolEvent(
        event_id="q",
        sequence=0,
        turn_id=1,
        name="get_user_details",
        duplicate_query=True,
        duplicate_of_event_id="earlier",
    )
    result = build_label_trace(
        inputs(1),
        [decision(0, refusal=True, target="A", operation="cancel")],
        [event],
        {"violating_turns": [], "attribution_complete": True},
    )
    assert result["turns"][0]["bad_turn"] is True
    assert result["turns"][0]["eligible_correct_refusal"] is False


def _transfer_runtime_record(*, valid=True, self_caused=False):
    from types import SimpleNamespace

    from test_transfer_policy_v2 import event, verdict

    from tau2_agentic_rl.reward.progress import build_progress_trace
    from tau2_agentic_rl.reward.score import build_reward_config, score_trajectory
    from tau2_agentic_rl.schemas import OfficialScores
    cfg = {"reward": {"mode": "turn_local_v1", "mandatory_policy_gate": False,
                      "task_safety_gate": False, "progress_version": "progress-v3",
                      "transfer_validation_version": "transfer-policy-v2"},
           "credit": {"version": "procredit-turn-v4"}}
    rule = {"required": True, "allowed": False}
    events = [event()]
    messages = [{"role": "assistant", "assistant_turn_id": i + 1, "content": "Response"}
                for i in range(3)]
    judge = verdict(valid=valid, self_caused_only=self_caused, policy_required=True)
    trace = build_progress_trace(task={}, messages=messages, prefix_lengths=[0, 1, 2, 3],
        events=events, required_actions=[], dependencies=[], transfer_rule=rule,
        initial_state_fingerprint="test", version="progress-v3")
    reward = score_trajectory(events=events, messages=messages, assistant_turns=3,
        required_actions=[], official=OfficialScores(reward=0), judge=judge,
        transfer_rule=rule, config=build_reward_config(cfg), progress_trace=trace)
    return SimpleNamespace(schema_version="2.0", custom_reward=reward, progress_trace=trace,
        response_turn_ids=[0, 1, 2], assistant_turns=3, trajectory_tokens=4,
        token_turns=[SimpleNamespace(prompt_token_ids=list(range(i + 1)),
            output_token_ids=[9], assistant_turn_index=i) for i in range(3)],
        trajectory_id="transfer", task_id="0", policy_version=0, judge_result=judge,
        environment_transcript=messages, tool_events=events,
        scoring_inputs={"reward_project_config": cfg, "judge": {"transfer_rule": rule}})


def test_training_reader_uses_same_guarded_transfer_completion():
    from tau2_agentic_rl.procredit_runtime import credit_from_record
    for valid, self_caused, expected in [(True, False, 1), (True, True, 0), (False, False, 0)]:
        rec = _transfer_runtime_record(valid=valid, self_caused=self_caused)
        assert rec.custom_reward.details["task_completion"] == expected
        assert credit_from_record(rec)["terminal_success"] == expected


def test_training_reader_rejects_stale_progress_version():
    import pytest

    from tau2_agentic_rl.procredit_runtime import credit_from_record
    rec = _transfer_runtime_record()
    rec.scoring_inputs["reward_project_config"]["reward"]["progress_version"] = "progress-v2"
    with pytest.raises(ValueError, match="progress version"):
        credit_from_record(rec)


def test_announced_unexecuted_transfer_is_explicit_policy_violation():
    from test_transfer_policy_v2 import inputs as transfer_inputs
    from test_transfer_policy_v2 import verdict

    from tau2_agentic_rl.judge.client import DeepSeekJudge
    from tau2_agentic_rl.schemas import JudgeCheck
    x = transfer_inputs()
    x["trajectory"]["tool_events"] = []
    x["trajectory"]["messages"] = [{"role": "assistant", "turn_idx": 2,
        "assistant_turn_id": 1, "content": "I will transfer you now."}]
    criterion = "task:policy:transfer_scope_and_message"
    x["mandatory_policy_checks"] = [{"criterion_id": criterion}]
    x["require_policy_attribution"] = True
    judge = verdict(applicable=True, valid=False, evidence_turn_ids=[])
    judge.mandatory_policy_checks = [JudgeCheck(criterion_id=criterion, applicable=True,
        passed=False, evidence_turn_ids=[2], violation_assistant_turn_ids=[1],
        short_reason="Assistant announced transfer without executing the tool.")]
    before = judge.model_dump()
    DeepSeekJudge._validate_requested_criteria(judge, x)
    assert judge.model_dump() == before


def test_unexecuted_transfer_applicability_cannot_invent_execution_or_blame():
    from copy import deepcopy

    import pytest
    from test_transfer_policy_v2 import inputs as transfer_inputs
    from test_transfer_policy_v2 import verdict

    from tau2_agentic_rl.judge.client import DeepSeekJudge
    from tau2_agentic_rl.schemas import JudgeCheck
    x = transfer_inputs()
    x["trajectory"]["tool_events"] = []
    x["trajectory"]["messages"] = [{"role": "assistant", "turn_idx": 2,
        "assistant_turn_id": 1, "content": "I will transfer you now."}]
    criterion = "task:policy:transfer_scope_and_message"
    x["mandatory_policy_checks"] = [{"criterion_id": criterion}]
    x["require_policy_attribution"] = True
    base = verdict(applicable=True, valid=False, evidence_turn_ids=[])
    base.mandatory_policy_checks = [JudgeCheck(criterion_id=criterion, applicable=True,
        passed=False, evidence_turn_ids=[2], violation_assistant_turn_ids=[1],
        short_reason="Unexecuted transfer announcement.")]
    cases = []
    for changes in [{"passed": True, "violation_assistant_turn_ids": []},
                    {"passed": True, "applicable": False, "violation_assistant_turn_ids": []},
                    {"violation_assistant_turn_ids": []}]:
        case = deepcopy(base)
        for key, value in changes.items():
            setattr(case.mandatory_policy_checks[0], key, value)
        cases.append(case)
    case = deepcopy(base)
    case.transfer_check.valid = True
    cases.append(case)
    case = deepcopy(base)
    case.transfer_check.evidence_turn_ids = [2]
    cases.append(case)
    for case in cases:
        with pytest.raises(ValueError):
            DeepSeekJudge._validate_requested_criteria(case, x)

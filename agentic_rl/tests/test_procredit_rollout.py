import asyncio
from types import SimpleNamespace

import pytest
from test_rollout_integration import minimal_loop

from tau2_agentic_rl.environment.tau2_gym import GymStep
from tau2_agentic_rl.procredit_runtime import credit_from_record
from tau2_agentic_rl.reward.progress import build_progress_trace
from tau2_agentic_rl.reward.score import build_reward_config
from tau2_agentic_rl.schemas import JudgeCheck, JudgeResult
from tau2_agentic_rl.scoring_retry import retry_scoring


def rollout_fixture(scratch_dir, *, policy_credit_v2=False, fail_after_delivery=False,
                    credit_version=None, judge_failures=0):
    loop, scope = minimal_loop(scratch_dir)
    cleaned, judge_calls = [], []
    loop.project["project"].update(tau2_commit="fixture", verl_commit="fixture")
    loop.project["reward"] = {"mode": "strict_progress_v1"}
    loop.project["judge"] = {"enabled": True}
    if credit_version:
        policy_credit_v2 = True
    if policy_credit_v2:
        loop.project["credit"] = {"version": credit_version or "procredit-turn-v2"}
    if credit_version == "procredit-turn-v4":
        loop.project["reward"].update(
            mode="turn_local_v1", mandatory_policy_gate=False, task_safety_gate=False,
        )
        loop.project["slot_recovery"] = {
            "max_resamples": 2, "max_scoring_retries": 2, "on_exhaustion": "quarantine_group",
        }
    loop.reward_config = build_reward_config(loop.project)
    loop.semantic, loop.transfer, loop.policy_rules = {"0": {}}, {"0": {}}, {"0": {}}
    loop.required_actions, loop.action_dependencies = {"0": []}, {}

    class Environment:
        task = {"id": "0", "evaluation_criteria": {"communicate_info": ["refund"]}}
        tool_schemas, tool_names = [], set()
        last_reward, info = 0, {}

        def __init__(self, **kwargs):
            self.messages = [{"role": "user", "content": "help"}]

        async def reset(self, **kwargs):
            return list(self.messages)

        async def step_text(self, content):
            assert content == "refund"
            self.messages += [
                {"role": "assistant", "content": content},
                {"role": "user", "content": "thanks"},
            ]
            if fail_after_delivery:
                raise RuntimeError("backend failed after delivery")
            return GymStep([self.messages[-1]], 0, False, {}, False, None, None)

        def full_trajectory(self):
            return self.messages

        def progress_initial_state(self):
            return "initial"

        async def force_cleanup_stop(self):
            cleaned.append(True)
            self.messages.append({"role": "assistant", "content": "cleanup"})

        def official_reward_payload(self):
            return 1, {
                "reward_basis": ["COMMUNICATE"],
                "communicate_checks": [{"met": True}],
            }

        def safe_db_hash(self):
            return "unchanged"

        initial_db_hash = safe_db_hash

        def user_prompt_hashes(self):
            return []

    generated = iter([[4, 9], [5, 9], [6]])

    async def generate(**kwargs):
        ids = next(generated)
        return SimpleNamespace(
            token_ids=ids, log_probs=[-1.0] * len(ids), extra_fields={}
        )

    async def parse(ids, *args):
        return (
            ("refund", [])
            if ids[0] == 4
            else ("mixed", [SimpleNamespace(name="noop", arguments="{}")])
        )

    async def render(*args, **kwargs):
        return [1, 2]

    async def bounded(messages, *args):
        return messages, [7], False

    async def snapshot(*args):
        return {}

    class Judge:
        async def evaluate(self, **kwargs):
            judge_calls.append(True)
            if len(judge_calls) <= judge_failures:
                raise RuntimeError("temporary Judge error")
            text = str(kwargs["trajectory"]["messages"])
            assert "cleanup" not in text and "unsent" not in text
            result = JudgeResult()
            if policy_credit_v2:
                result.mandatory_policy_checks = [JudgeCheck(
                    criterion_id="mixed", passed=False, evidence_turn_ids=[2],
                    violation_assistant_turn_ids=[2], short_reason="mixed text/tool action",
                )]
            return result, "", "", ""

    loop.tokenizer = SimpleNamespace(
        eos_token_id=9,
        decode=lambda ids: {
            (4, 9): "refund<|im_end|>",
            (4,): "refund",
            (5, 9): 'mixed<tool_call>{"name":"noop","arguments":{}}</tool_call>',
            (6,): "unsent",
        }[tuple(ids)],
    )
    loop._render_full_chat, loop._bounded_environment_messages = render, bounded
    loop.tool_parser = SimpleNamespace(extract_tool_calls=parse)
    loop.server_manager = SimpleNamespace(generate=generate)
    loop.shared_budget = SimpleNamespace(acall=snapshot)
    loop.judge = Judge()
    scope["Tau2GymAdapter"] = Environment

    def local_verifier(**kwargs):
        return build_progress_trace(
            **kwargs,
            evaluator=lambda task, messages: {
                "db": None,
                "communicate": [
                    any(
                        m["role"] == "assistant" and m.get("content") == "refund"
                        for m in messages
                    )
                ],
            },
        )

    scope["build_progress_trace"] = local_verifier
    if fail_after_delivery:
        with pytest.raises(RuntimeError, match="tau2_text_step; audit record saved") as caught:
            asyncio.run(loop._run_trajectory({}, extra_info={"task_id": "0"}))
        assert str(caught.value.__cause__) == "backend failed after delivery"
        assert cleaned == [True]
        assert not judge_calls
        return loop, None, next(loop.store.records())
    output = asyncio.run(loop._run_trajectory({}, extra_info={"task_id": "0"}))
    return loop, output, next(loop.store.records())


def test_v2_partial_environment_failure_keeps_original_failure_and_audit(scratch_dir):
    _, _, record = rollout_fixture(
        scratch_dir, policy_credit_v2=True, fail_after_delivery=True
    )
    assert record.metadata["failure_phase"] == "tau2_text_step"
    assert record.custom_reward is None and record.progress_trace is None
    assert [m.get("content") for m in record.environment_transcript] == [
        "help", "refund", "thanks"
    ]
    assert record.token_turns[0].output_token_ids == [4, 9]


def test_v3_actor_frozen_retry_and_cost_tamper_detection(scratch_dir):
    loop, output, record = rollout_fixture(scratch_dir, credit_version="procredit-turn-v3")
    assert record.progress_trace["version"] == "progress-v2"
    assert record.custom_reward.train_reward == 0
    assert output.extra_fields["procredit"]["process_credit"]["turn_costs"] == pytest.approx([0, .1, 0])
    expected = record.custom_reward.model_dump()
    record.custom_reward = None
    record.metadata["failure_phase"] = "judge"
    assert asyncio.run(retry_scoring(record, loop.judge, loop.store))
    assert record.custom_reward.model_dump() == expected
    record.custom_reward.details["process_credit"]["turn_costs"][0] = .1
    with pytest.raises(ValueError, match="process"):
        credit_from_record(record)


def test_actor_records_delivered_prefixes_and_maps_every_policy_token(scratch_dir):
    _, output, record = rollout_fixture(scratch_dir)
    assert record.schema_version == "2.0"
    assert record.progress_trace["prefix_lengths"] == [1, 3, 3, 3]
    assert record.progress_trace["phi"] == [0, 1, 1, 1]
    assert record.response_turn_ids == [0, 0, -1, 1, 1, -1, 2]
    assert output.response_ids == [4, 9, 7, 5, 9, 7, 6]
    assert output.response_logprobs == [-1, -1, 0, -1, -1, 0, -1]
    assert output.reward_score == pytest.approx(1.025)
    assert output.extra_fields["procredit"]["phi"] == [0, 1, 1, 1]
    assert "cleanup" not in str(record.scoring_inputs["progress_inputs"])
    assert record.official_scores.reward == 0


def test_frozen_judge_retry_preserves_new_reward_and_trace(scratch_dir):
    loop, _, record = rollout_fixture(scratch_dir)
    expected = record.custom_reward.train_reward
    record.custom_reward = None
    record.metadata["failure_phase"] = "judge"
    assert asyncio.run(retry_scoring(record, loop.judge, loop.store))
    assert record.custom_reward.train_reward == expected


@pytest.mark.parametrize("change", ["turn_map", "reward", "checkset"])
def test_credit_record_rejects_changes_to_bound_scoring_evidence(scratch_dir, change):
    _, output, record = rollout_fixture(scratch_dir)
    assert credit_from_record(record) == output.extra_fields["procredit"]
    if change == "turn_map":
        record.response_turn_ids[0] = 1
    elif change == "reward":
        record.custom_reward.train_reward = 0.9
    else:
        record.custom_reward.details["checkset_fingerprint"] = "other"
    with pytest.raises(ValueError):
        credit_from_record(record)


def test_v2_actor_scoring_retry_and_group_credit_preserve_local_policy_signal(scratch_dir):
    from copy import deepcopy

    from tau2_agentic_rl.advantages import CreditConfig, compute_group_credit

    loop, output, record = rollout_fixture(scratch_dir, policy_credit_v2=True)
    assert output.reward_score == record.custom_reward.strict_success == 0
    assert record.custom_reward.details["policy_credit"]["violating_turns"] == [1]
    assert record.environment_transcript[1]["assistant_turn_id"] == 1
    assert record.scoring_inputs["reward_project_config"]["credit"]["version"] == "procredit-turn-v2"
    rows = [deepcopy(output.extra_fields["procredit"]) for _ in range(8)]
    for i, item in enumerate(rows):
        item["trajectory_id"] = f"sample-{i}"
    credit = compute_group_credit(rows, CreditConfig(version="procredit-turn-v2"))
    assert credit["has_signal"]
    assert credit["trajectories"][0]["turn_advantages"][0] > 0
    assert credit["trajectories"][0]["turn_advantages"][1] < 0
    expected = record.custom_reward.model_dump()
    record.custom_reward = None
    record.metadata["failure_phase"] = "judge"
    assert asyncio.run(retry_scoring(record, loop.judge, loop.store))
    assert record.custom_reward.model_dump() == expected
    record.custom_reward.details["policy_credit"]["violating_turns"] = [0]
    with pytest.raises(ValueError, match="policy"):
        credit_from_record(record)

"""Production loop + simulated transport/environment, not a veRL dependency test."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from tau2_agentic_rl.budget import ContextBudget
from tau2_agentic_rl.concurrency import QueueWaitError, SharedBudget
from tau2_agentic_rl.environment.tau2_gym import GymStep
from tau2_agentic_rl.reward.score import RewardConfig
from tau2_agentic_rl.schemas import JudgeCheck, JudgeResult
from tau2_agentic_rl.storage import TrajectoryStore


def production_loop_class():
    path = Path(__file__).parents[1] / "src/tau2_agentic_rl/agent_loop/airline.py"
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    # Only replace external veRL transport types. Execute ALL project code and
    # the complete production run method, rather than copying its algorithm.
    parsed.body = [
        node
        for node in parsed.body
        if not (
            isinstance(node, ast.ImportFrom) and (node.module or "").startswith("verl.")
        )
    ]
    scope = {
        "__name__": "loop_fixture",
        "AgentLoopBase": object,
        "AgentLoopOutput": SimpleNamespace,
        "AgentLoopMetrics": SimpleNamespace,
        "register": lambda name: lambda cls: cls,
        "OpenAIFunctionToolSchema": lambda **kwargs: kwargs,
    }
    exec(compile(parsed, str(path), "exec"), scope)
    return scope["Tau2AirlineAgentLoop"], scope


def test_truncated_output_not_delivered_or_judged_and_full_observation_preserved(
    scratch_dir,
):
    cls, scope = production_loop_class()
    full_observation = "tool-independent user detail " * 2000

    class Environment:
        policy, task, tool_schemas, tool_names = "policy", {}, [], set()

        def __init__(self, **kwargs):
            self.messages = [{"role": "user", "content": "Explain refunds"}]

        async def reset(self, **kwargs):
            return list(self.messages)

        async def step_text(self, content):
            assert content == "Let me check.<|im_end|>"
            self.messages += [
                {"role": "assistant", "content": content},
                {"role": "user", "content": full_observation},
            ]
            return GymStep(
                messages=[self.messages[-1]],
                reward=0,
                terminated=False,
                info={},
                db_changed=False,
                tool_success=None,
                tool_result=None,
            )

        def full_trajectory(self):
            return self.messages

        async def force_cleanup_stop(self):
            self.messages.append({"role": "assistant", "content": "###STOP###"})

        def official_reward_payload(self):
            return 1, {
                "reward_basis": ["DB"],
                "db_check": {"db_match": True, "db_reward": 1},
            }

        def safe_db_hash(self):
            return "completed-state"

        def initial_db_hash(self):
            return "initial-state"

        def user_prompt_hashes(self):
            return []

    async def template(messages, **kwargs):
        return [1, 2]

    async def parse(*args):
        return "Let me check.", []

    async def bounded(*args):
        return [{"role": "user", "content": "[truncated]"}], [5, 6], True

    generated = iter([[4, 9], [3]])

    async def generate(**kwargs):
        tokens = next(generated)
        return SimpleNamespace(
            token_ids=tokens, log_probs=[-1.0] * len(tokens), extra_fields={}
        )

    async def snapshot(*args):
        return {"peak_active_trajectories": 1}

    class Judge:
        async def evaluate(self, **inputs):
            transcript = inputs["trajectory"]["messages"]
            assert transcript[-1]["content"] == full_observation
            assert "Unsent refund explanation" not in str(transcript)
            assert "###STOP###" not in str(transcript)
            return (
                JudgeResult(
                    semantic_checks=[JudgeCheck(criterion_id="explain", passed=False)]
                ),
                "raw",
                "hash",
                "cache",
            )

    loop = cls.__new__(cls)
    loop.project = {
        "user_simulator": {"model": "fake", "temperature": 0},
        "outputs": {"user_cache": "cache"},
        "rollout": {},
        "project": {
            "annotation_version": "v1",
            "reward_version": "v1",
            "tau2_commit": "fixture",
            "verl_commit": "fixture",
        },
    }
    loop.root, loop.hard_turn_limit = scratch_dir, 24
    loop.tokenizer = SimpleNamespace(
        eos_token_id=9,
        decode=lambda ids: (
            "Let me check.<|im_end|>" if ids[-1] == 9 else "Unsent refund explanation"
        ),
    )
    loop.tool_parser = SimpleNamespace(extract_tool_calls=parse)
    loop._render_full_chat = template
    loop.rollout_config = SimpleNamespace(prompt_length=8192)
    loop._bounded_environment_messages = bounded
    loop.server_manager = SimpleNamespace(generate=generate)
    loop.shared_budget = SimpleNamespace(acall=snapshot)
    loop.budget = ContextBudget()
    loop.judge, loop.reward_config = Judge(), RewardConfig()
    loop.store = TrajectoryStore(
        scratch_dir / "records", attach_evaluation_identity=False
    )
    loop.semantic = {"0": {"semantic_checks": [{"criterion_id": "explain"}]}}
    loop.transfer, loop.policy_rules = {"0": {}}, {"0": {}}
    loop.required_actions, loop.action_dependencies = {"0": []}, {}
    scope["Tau2GymAdapter"] = Environment
    output = asyncio.run(loop._run_trajectory({}, extra_info={"task_id": "0"}))
    record = next(loop.store.records())
    assert (
        record.termination_reason
        == output.extra_fields["termination_reason"]
        == "generation_truncated"
    )
    assert record.official_scores.reward == 0
    assert 0 < record.custom_reward.train_reward < 1
    assert record.messages[-1]["content"] == "Unsent refund explanation"
    assert record.messages[-2]["content"] == "[truncated]"
    assert record.environment_transcript[-1]["content"] == full_observation
    assert record.metadata["initial_prompt"]["initial_prompt_tokens"] == 2


def minimal_loop(scratch_dir):
    cls, scope = production_loop_class()
    loop = cls.__new__(cls)
    loop.root, loop.hard_turn_limit = scratch_dir, 24
    loop.project = {
        "user_simulator": {"model": "fake", "temperature": 0},
        "outputs": {"user_cache": "cache"},
        "rollout": {
            "max_active_trajectories": 1,
            "user_api_max_inflight": 1,
            "judge_api_max_inflight": 1,
            "queue_timeout_seconds": 0.001,
        },
        "project": {"annotation_version": "v1", "reward_version": "v1"},
    }
    loop.store = TrajectoryStore(
        scratch_dir / "records", attach_evaluation_identity=False
    )
    loop.budget = ContextBudget()
    loop.rollout_config = SimpleNamespace(prompt_length=8192)
    return loop, scope


def test_full_initial_prompt_rejected_before_generation_and_audited(scratch_dir):
    loop, scope = minimal_loop(scratch_dir)
    cleaned = []

    class Environment:
        policy, tool_schemas = "full policy", []

        def __init__(self, **kwargs):
            pass

        async def reset(self, **kwargs):
            return [{"role": "user", "content": "large input"}]

        async def force_cleanup_stop(self):
            cleaned.append(True)

        def initial_db_hash(self):
            return "initial"

        def safe_db_hash(self):
            return "final"

    async def full(messages, **kwargs):
        assert messages[0]["content"].startswith("full policy")
        return list(range(8193))

    scope["Tau2GymAdapter"] = Environment
    loop._render_full_chat = full
    # No server_manager: reaching model generation would itself fail the test.
    with pytest.raises(RuntimeError, match="prompt initialization"):
        asyncio.run(loop._run_trajectory({}, extra_info={"task_id": "23"}))
    record = next(loop.store.records())
    assert cleaned == [True]
    assert record.trajectory_tokens == 8193
    assert record.metadata["initial_prompt"]["initial_prompt_tokens"] == 8193
    assert record.metadata["initial_prompt"]["task_id"] == "23"
    assert record.metadata["failure_phase"] == "prompt_initialization"
    assert record.assistant_turns == 0


def test_queue_failure_has_trajectory_id_and_does_not_create_environment(scratch_dir):
    loop, scope = minimal_loop(scratch_dir)
    local = SharedBudget(
        {"trajectories": 1, "user_api": 1, "judge_api": 1}, queue_timeout_seconds=0.001
    )
    scope["SharedBudget"] = lambda *args, **kwargs: local

    async def check():
        async with local.aslot("trajectories"):
            with pytest.raises(QueueWaitError):
                await loop.run(
                    {}, extra_info={"task_id": "23", "evaluation_sample_index": 2}
                )
            assert local.call("snapshot")["active"]["trajectories"] == 1

    asyncio.run(check())
    record = next(loop.store.records())
    assert record.trajectory_id and record.task_id == "23"
    assert record.metadata["failure_phase"] == "trajectory_queue"
    assert record.metadata["evaluation_sample_index"] == 2
    assert record.messages == [] and record.assistant_turns == 0


def test_repeated_cancellation_retains_lease_until_background_work_finishes(
    scratch_dir,
):
    import threading

    loop, scope = minimal_loop(scratch_dir)
    local = SharedBudget({"trajectories": 1, "user_api": 1, "judge_api": 1})
    scope["SharedBudget"] = lambda *args, **kwargs: local
    finish = threading.Event()

    async def check():
        started = asyncio.Event()

        async def trajectory(*args, **kwargs):
            started.set()
            await asyncio.to_thread(finish.wait)
            kwargs["lease"].progress()
            return "finished"

        loop._run_trajectory = trajectory
        task = asyncio.create_task(loop.run({}, extra_info={"task_id": "0"}))
        try:
            await started.wait()
            for _ in range(2):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
                assert local.call("snapshot")["active"]["trajectories"] == 1
        finally:
            finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert local.call("snapshot")["active"]["trajectories"] == 0

    asyncio.run(check())

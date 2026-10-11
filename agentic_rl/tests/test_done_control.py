"""Tau2 stop control is not an environment tool result or a successful write."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from test_mixed_tool_execution import Message, native_adapter
from test_rollout_integration import minimal_loop

from tau2_agentic_rl.reward.process_penalty import compute_process_penalty
from tau2_agentic_rl.reward.progress import build_progress_trace
from tau2_agentic_rl.reward.required_actions import evaluate_required_actions
from tau2_agentic_rl.reward.score import RewardConfig, build_reward_config
from tau2_agentic_rl.schemas import JudgeResult


def stop_adapter(monkeypatch, *, reason="agent_stop", terminated=True):
    adapter, backend, original, executed = native_adapter(monkeypatch)
    agent = adapter.env._agent
    ordinary_action = agent.set_action
    info = {
        "task": {"id": "5"},
        "tools": [SimpleNamespace(name=name, openai_schema={
            "type": "function", "function": {
                "name": name, "parameters": {
                    "type": "object", "properties": {}, "additionalProperties": False,
                },
            },
        }) for name in ("done", "write")],
    }

    def act(action):
        if not any(call.name == "done" for call in action.tool_calls):
            ordinary_action(action)
            return
        # Pinned GymAgent._check_if_stop_toolcall replaces the WHOLE batch.
        action.content, action.tool_calls = "###STOP###", None
        agent.observation.append(action)
        adapter.env._orchestrator.trajectory.append(action)
        if terminated:
            adapter.env._simulation_done.set()
            info["simulation_run"] = json.dumps({"termination_reason": reason})

    agent.set_action = act
    adapter.env._get_info = lambda: dict(info)
    adapter.info = info
    adapter._initial_db_hash = "0"
    return adapter, backend, original, executed


def test_single_done_stops_without_fabricating_tool_message_or_success(monkeypatch):
    adapter, backend, original, executed = stop_adapter(monkeypatch)
    step = adapter._step_tools_sync([{"id": "stop", "name": "done", "arguments": {}}])
    assert step.terminated and adapter.tau2_termination_reason() == "agent_stop"
    assert step.messages == [] and executed == []
    assert step.db_changed is False and step.tool_success is None
    assert len(step.tool_results) == 1
    control = step.tool_results[0]
    assert (control.call_id, control.name, control.arguments) == ("stop", "done", {})
    assert control.execution_status == "termination_control"
    assert control.success is False and control.db_changed is False
    assert control.result is None
    assert backend.get_response is original
    assert [m["role"] for m in adapter.full_trajectory()] == ["assistant"]
    assert adapter.full_trajectory()[0]["content"] == "###STOP###"


@pytest.mark.parametrize("reason,terminated", [
    ("user_stop", True), ("max_steps", True), ("agent_error", True),
    ("agent_stop", False),
])
def test_done_without_confirmed_agent_stop_still_requires_results(monkeypatch, reason, terminated):
    adapter, backend, original, _ = stop_adapter(monkeypatch, reason=reason, terminated=terminated)
    with pytest.raises(RuntimeError, match="0 results for 1 tool calls"):
        adapter._step_tools_sync([{"id": "stop", "name": "done", "arguments": {}}])
    assert backend.get_response is original


def test_normal_tool_missing_result_still_fails_even_at_agent_stop(monkeypatch):
    adapter, _, _, executed = stop_adapter(monkeypatch)
    adapter.env._agent.set_action = lambda action: adapter.env._simulation_done.set()
    adapter.env._get_info = lambda: {"simulation_run": {"termination_reason": "agent_stop"}}
    with pytest.raises(RuntimeError, match="0 results for 1 tool calls"):
        adapter._step_tools_sync([{"id": "write", "name": "write", "arguments": {}}])
    assert executed == []


@pytest.mark.parametrize("names", [("done", "write"), ("write", "done"), ("done", "done")])
def test_adapter_rejects_mixed_stop_before_delivering_any_call(monkeypatch, names):
    adapter, backend, original, executed = stop_adapter(monkeypatch)
    with pytest.raises(ValueError, match="done.*alone"):
        adapter._step_tools_sync([
            {"id": str(i), "name": name, "arguments": {}} for i, name in enumerate(names)
        ])
    assert executed == [] and adapter.full_trajectory() == []
    assert not adapter.env._simulation_done.is_set()
    assert backend.get_response is original


@pytest.mark.parametrize("first_names", [None, ("done", "write"), ("write", "done"), ("done", "done")])
@pytest.mark.parametrize("procredit", [False, True])
def test_rollout_recovers_from_mixed_stop_and_preserves_control_semantics(
    monkeypatch, scratch_dir, first_names, procredit,
):
    adapter, _, _, executed = stop_adapter(monkeypatch)
    loop, scope = minimal_loop(scratch_dir)
    user = Message(role="user", content="Finish the task.")
    adapter.env._orchestrator.trajectory.append(user)
    adapter.env._agent.observation.append(user)
    adapter._observation_count = 1
    reward_info = {"reward_basis": ["DB"], "db_check": {"db_match": False, "db_reward": 0}}
    adapter.official_reward_payload = lambda: (0, reward_info)

    async def reset(**kwargs):
        return [{"role": "user", "content": user.content}]

    adapter.reset = reset
    batches = ([first_names] if first_names else []) + [("done",)]
    cursor = iter(range(len(batches)))

    async def generate(**kwargs):
        return SimpleNamespace(token_ids=[4 + next(cursor), 9], log_probs=[-0.2, -0.1], extra_fields={"global_steps": 0})

    async def parse(ids, *args):
        return "", [SimpleNamespace(name=name, arguments="{}") for name in batches[ids[0] - 4]]

    def decode(ids):
        return "".join('<tool_call>' + json.dumps({"name": name, "arguments": {}}) + '</tool_call>'
                       for name in batches[ids[0] - 4]) + "<|im_end|>"

    async def render(*args, **kwargs):
        return [1, 2]

    async def bounded(messages, *args):
        return messages, [7], False

    async def snapshot(*args):
        return {}

    loop.tokenizer = SimpleNamespace(eos_token_id=9, decode=decode)
    loop.tool_parser = SimpleNamespace(extract_tool_calls=parse)
    loop.server_manager = SimpleNamespace(generate=generate)
    loop._render_full_chat, loop._bounded_environment_messages = render, bounded
    loop.shared_budget = SimpleNamespace(acall=snapshot)
    loop.judge, loop.reward_config = None, RewardConfig()
    loop.project["judge"] = {"enabled": False}
    loop.semantic, loop.transfer, loop.policy_rules = {"5": {}}, {"5": {}}, {"5": {}}
    loop.required_actions, loop.action_dependencies = {"5": []}, {}
    loop.project["project"].update(tau2_commit="fixture", verl_commit="fixture")
    scope["Tau2GymAdapter"] = lambda **kwargs: adapter
    if procredit:
        loop.project["reward"] = {
            "mode": "turn_local_v1", "mandatory_policy_gate": False, "task_safety_gate": False,
        }
        loop.project["credit"] = {"version": "procredit-turn-v4"}
        loop.project["judge"] = {"enabled": True}
        loop.reward_config = build_reward_config(loop.project)
        adapter.progress_initial_state = lambda: "initial"

        async def judge(**kwargs):
            return JudgeResult(), "", "", ""

        loop.judge = SimpleNamespace(evaluate=judge)
        scope["build_progress_trace"] = lambda **kwargs: build_progress_trace(
            **kwargs, evaluator=lambda task, messages: {"db": False, "communicate": []},
        )

    output = asyncio.run(loop._run_trajectory({}, extra_info={"task_id": "5"}))
    record = next(loop.store.records())
    assert record.termination_reason == "agent_stop"
    assert record.metadata.get("failure_phase") is None
    assert record.assistant_turns == len(batches)
    assert len(record.token_turns) == len(batches)
    assert [p for p, m in zip(output.response_logprobs, output.response_mask, strict=True) if m] == [-0.2, -0.1] * len(batches)
    assert executed == []
    assert [m["role"] for m in record.environment_transcript] == ["user", "assistant"]
    assert record.environment_transcript[-1]["content"] == "###STOP###"
    control = record.tool_events[-1]
    assert control.execution_status == "termination_control"
    assert control.success is False and control.db_effect is False
    assert control.error_kind is None and not control.unchanged_retry
    assert compute_process_penalty([control], len(batches)).penalty == 0
    if procredit:
        assert record.custom_reward.details["process_credit"]["turn_costs"][-1] == 0
        assert output.extra_fields["procredit"]["process_credit"]["turn_costs"][-1] == 0
    assert evaluate_required_actions([
        {"action_id": "not_a_real_action", "name": "done", "arguments": {}},
    ], [control]).completed_action_ids == []
    if first_names:
        assert record.tool_events[0].error_kind == "multiple_tool_calls"
        assert not any(event.success for event in record.tool_events)
        errors = [m["content"] for m in record.messages if m["role"] == "tool"]
        assert len(errors) == 1 and "done" in errors[0] and "alone" in errors[0]

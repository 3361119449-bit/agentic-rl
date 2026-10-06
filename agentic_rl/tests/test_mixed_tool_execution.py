import json
import sys
import threading
from types import SimpleNamespace

import pytest

from tau2_agentic_rl.environment.tau2_gym import Tau2GymAdapter


class Message(SimpleNamespace):
    def model_dump(self, **kwargs):
        return {key: (value if not isinstance(value, list) else [
            item.model_dump(**kwargs) if hasattr(item, "model_dump") else item for item in value
        ]) for key, value in vars(self).items()}


def native_adapter(monkeypatch, runtime_error=False):
    monkeypatch.setitem(sys.modules, "tau2.data_model.message", SimpleNamespace(
        AssistantMessage=Message, ToolCall=Message, ToolMessage=Message,
    ))
    executed = []
    version = [0]

    def respond(call):
        executed.append(call.id)
        error = runtime_error and call.id == "valid0"
        if not error:
            version[0] += 1
        return Message(id=call.id, role="tool", content="failed" if error else "ok",
                       requestor="assistant", error=error)

    backend = SimpleNamespace(get_response=respond, get_db_hash=lambda: str(version[0]))
    agent = SimpleNamespace(observation=[], is_agent_turn=True)
    transcript = []

    def act(action):
        agent.observation.append(action)
        transcript.append(action)
        # Same ordered get_response loop as pinned Tau2's orchestrator.
        for call in action.tool_calls:
            result = backend.get_response(call)
            agent.observation.append(result)
            transcript.append(result)

    agent.set_action = act
    adapter = Tau2GymAdapter(task_id="0", user_model="fake")
    adapter.env = SimpleNamespace(
        _orchestrator=SimpleNamespace(environment=backend, trajectory=transcript),
        _agent=agent, _lock=threading.Lock(), _simulation_done=threading.Event(),
        _get_reward=lambda: (1, {}), _get_info=lambda: {},
    )
    return adapter, backend, respond, executed


@pytest.mark.parametrize("bad_position", [0, 1, 2])
@pytest.mark.parametrize("error_kind,arguments,raw", [
    ("schema_invalid", {}, "[]"),
    ("schema_invalid", {"value": "bad"}, '{"value":"bad"}'),
    ("unknown_tool", {"value": 0}, '{"value":0}'),
])
def test_native_mixed_batch_executes_valid_calls_and_keeps_all_result_ids(
    monkeypatch, bad_position, error_kind, arguments, raw,
):
    adapter, backend, original, executed = native_adapter(monkeypatch)
    calls = [
        {"id": f"valid{index}", "name": "write", "arguments": {"value": index}}
        for index in range(2)
    ]
    calls.insert(bad_position, {
        "id": "invalid", "name": "unknown" if error_kind == "unknown_tool" else "write",
        "arguments": arguments, "raw_arguments": raw,
        "validation_error": {"kind": error_kind, "detail": "rejected arguments"},
    })
    step = adapter._step_tools_sync(calls)
    assert executed == ["valid0", "valid1"]
    assert [result.call_id for result in step.tool_results] == [call["id"] for call in calls]
    assert [result.success for result in step.tool_results] == [call["id"] != "invalid" for call in calls]
    assert [result.db_changed for result in step.tool_results] == [call["id"] != "invalid" for call in calls]
    assert [message["tool_call_id"] for message in step.messages] == [call["id"] for call in calls]
    observation = step.messages[bad_position]
    assert observation["error"] is True
    assert json.loads(observation["content"])["type"] == error_kind
    assert backend.get_response is original
    assert step.db_changed is True and step.tool_success is False
    transcript = adapter.full_trajectory()
    assert len(transcript) == 4
    assert [call["id"] for call in transcript[0]["tool_calls"]] == [call["id"] for call in calls]
    assert transcript[0]["raw_data"]["rejected_tool_calls"][0]["raw_arguments"] == raw


def test_returned_execution_error_does_not_block_another_valid_call(monkeypatch):
    adapter, backend, original, executed = native_adapter(monkeypatch, runtime_error=True)
    step = adapter._step_tools_sync([
        {"id": "valid0", "name": "write", "arguments": {"value": 0}},
        {"id": "bad", "name": "write", "arguments": {},
         "validation_error": {"kind": "schema_invalid", "detail": "missing value"}},
        {"id": "valid1", "name": "write", "arguments": {"value": 1}},
    ])
    assert executed == ["valid0", "valid1"]
    assert [result.success for result in step.tool_results] == [False, False, True]
    assert [result.db_changed for result in step.tool_results] == [False, False, True]
    assert backend.get_response is original

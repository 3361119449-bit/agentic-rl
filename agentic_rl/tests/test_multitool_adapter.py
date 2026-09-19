import threading
from types import SimpleNamespace

import pytest

message_module = pytest.importorskip("tau2.data_model.message")
ToolMessage = message_module.ToolMessage

from tau2_agentic_rl.environment.tau2_gym import Tau2GymAdapter


class BackendEnvironment:
    def __init__(self):
        self.version = 0

    def get_db_hash(self):
        return str(self.version)

    def get_response(self, tool_call):
        if tool_call.name == "write":
            self.version += 1
        return ToolMessage(
            id=tool_call.id,
            role="tool",
            content=f"{tool_call.name}:{tool_call.arguments['value']}",
            requestor="assistant",
            error=False,
        )


class FakeAgent:
    def __init__(self, backend):
        self.backend = backend
        self.observation = []
        self.is_agent_turn = True

    def set_action(self, action):
        self.is_agent_turn = False
        self.observation.append(action)
        for call in action.tool_calls:
            self.observation.append(self.backend.get_response(call))
        self.is_agent_turn = True


class FakeGym:
    def __init__(self):
        backend = BackendEnvironment()
        self._orchestrator = SimpleNamespace(environment=backend)
        self._agent = FakeAgent(backend)
        self._simulation_done = threading.Event()
        self._lock = threading.Lock()

    def _get_reward(self):
        return 0.0, {}

    def _get_info(self):
        return {"simulation_run": {}, "tools": []}


def test_step_tools_returns_every_result_and_preserves_call_ids():
    adapter = Tau2GymAdapter(task_id="0", user_model="fake")
    adapter.env = FakeGym()
    adapter.info = {"simulation_run": {}, "tools": []}

    step = adapter._step_tools_sync(
        [
            {"id": "call_a", "name": "read", "arguments": {"value": 1}},
            {"id": "call_b", "name": "write", "arguments": {"value": 2}},
        ]
    )

    assert [item["tool_call_id"] for item in step.messages] == ["call_a", "call_b"]
    assert [item.result for item in step.tool_results] == ["read:1", "write:2"]
    assert [item.db_changed for item in step.tool_results] == [False, True]
    assert step.tool_success is True
    assert step.db_changed is True

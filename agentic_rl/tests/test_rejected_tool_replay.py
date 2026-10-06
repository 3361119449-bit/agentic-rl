import json
import sys
from types import SimpleNamespace

import pytest
from test_mixed_tool_execution import native_adapter


class ReplayEnvironment:
    def __init__(self, **kwargs):
        self.executed = []

    def get_response(self, call):
        self.executed.append(call.id)
        return SimpleNamespace(content="ok", error=False)

    def set_state(self, initialization_data, initialization_actions, message_history, strict=True):
        for message in message_history:
            calls = getattr(message, "tool_calls", None)
            if calls:
                for call in calls:
                    expected = next(item for item in message_history if getattr(item, "id", None) == call.id)
                    actual = self.get_response(call)
                    if strict and actual.content != expected.content:
                        raise ValueError("strict replay mismatch")


def transcript(monkeypatch):
    adapter, _, _, _ = native_adapter(monkeypatch)
    adapter._step_tools_sync([
        {"id": "valid0", "name": "write", "arguments": {"value": 1}},
        {"id": "rejected", "name": "write", "arguments": {},
         "raw_arguments": "[]", "validation_error": {"kind": "schema_invalid", "detail": "missing value"}},
        {"id": "valid1", "name": "write", "arguments": {"value": 2}},
    ])
    return adapter.env._orchestrator.trajectory


def test_rejected_write_replay_never_executes_it_and_keeps_strict_checking(monkeypatch):
    from tau2_agentic_rl.environment.replay import rejection_aware_constructor

    history = transcript(monkeypatch)
    before = [item.model_dump() for item in history]
    env = rejection_aware_constructor(ReplayEnvironment)()
    original = env.get_response
    env.set_state(None, None, history, strict=True)
    assert env.executed == ["valid0", "valid1"]
    assert env.get_response == original
    assert [item.model_dump() for item in history] == before
    history[1].content = "tampered successful output"
    with pytest.raises(ValueError, match="strict replay mismatch"):
        env.set_state(None, None, history, strict=True)
    assert env.get_response == original


@pytest.mark.parametrize("change", ["claimed_success", "changed_error", "unknown_id"])
def test_replay_rejects_inconsistent_no_execution_evidence(monkeypatch, change):
    from tau2_agentic_rl.environment.replay import rejection_aware_constructor

    history = transcript(monkeypatch)
    if change == "claimed_success":
        history[2].error = False
    elif change == "changed_error":
        history[2].content = "different error"
    else:
        history[0].raw_data["rejected_tool_calls"][0]["id"] = "absent"
    env = rejection_aware_constructor(ReplayEnvironment)()
    with pytest.raises(ValueError, match="rejected tool"):
        env.set_state(None, None, history, strict=True)
    assert env.executed == []


def test_final_gym_reward_uses_scoped_replay_without_mutating_official_registry(monkeypatch):
    from tau2_agentic_rl.environment.cached_user import build_cached_agent_gym_env

    history = transcript(monkeypatch)
    namespace = {"registry": SimpleNamespace(get_env_constructor=lambda domain: ReplayEnvironment),
                 "SimpleNamespace": SimpleNamespace, "json": json}
    exec("""
def evaluate_simulation(simulation, task, evaluation_type, solo_mode, domain):
    env = registry.get_env_constructor(domain)(solo_mode=solo_mode)
    env.set_state(None, None, simulation.messages, strict=True)
    return SimpleNamespace(reward=len(env.executed), model_dump_json=lambda **kwargs: "{}")
""", namespace)
    evaluate = namespace["evaluate_simulation"]
    original_registry = namespace["registry"]

    class Gym:
        def __init__(self, **kwargs):
            self._simulation_run = SimpleNamespace(messages=history)
            self.domain, self.solo_mode = "airline", False

        def _get_task(self):
            return {}

        def _get_reward(self):
            # The pinned parent calls the unmodified official evaluator.
            info = evaluate(self._simulation_run, {}, "all", False, "airline")
            return info.reward, info.model_dump_json()

    monkeypatch.setitem(sys.modules, "tau2.gym.gym_agent", SimpleNamespace(AgentGymEnv=Gym))
    monkeypatch.setitem(sys.modules, "tau2.user.user_simulator", SimpleNamespace(UserSimulator=object))
    monkeypatch.setitem(sys.modules, "tau2.evaluator.evaluator", SimpleNamespace(
        evaluate_simulation=evaluate, EvaluationType=SimpleNamespace(ALL="all"),
    ))
    env = build_cached_agent_gym_env(user_cache_dir="unused")
    reward, _ = env._get_reward()
    assert reward == 2
    assert evaluate.__globals__["registry"] is original_registry
    with pytest.raises(ValueError, match="strict replay mismatch"):
        evaluate(env._simulation_run, {}, "all", False, "airline")
    assert env._simulation_run.messages is history


def test_progress_prefix_uses_guarded_db_replay_and_full_communication_history(monkeypatch):
    import pydantic

    from tau2_agentic_rl.reward.progress import evaluate_tau2_prefix

    history = transcript(monkeypatch)
    replayed, communicated = [], []

    def env_reward(**kwargs):
        env = kwargs["environment_constructor"]()
        env.set_state(None, None, kwargs["full_trajectory"], strict=kwargs["strict_replay"])
        replayed.extend(env.executed)
        return SimpleNamespace(db_check=SimpleNamespace(db_reward=1.0))

    def comm_reward(**kwargs):
        communicated.extend(kwargs["full_trajectory"])
        return SimpleNamespace(communicate_checks=[SimpleNamespace(met=True)])

    monkeypatch.setattr(pydantic, "TypeAdapter", lambda _: SimpleNamespace(validate_python=lambda values: history))
    monkeypatch.setitem(sys.modules, "tau2.data_model.message", SimpleNamespace(Message=SimpleNamespace))
    monkeypatch.setitem(sys.modules, "tau2.data_model.tasks", SimpleNamespace(
        Task=SimpleNamespace(model_validate=lambda task: task),
    ))
    monkeypatch.setitem(sys.modules, "tau2.evaluator.evaluator_env", SimpleNamespace(
        EnvironmentEvaluator=SimpleNamespace(calculate_reward=env_reward),
    ))
    monkeypatch.setitem(sys.modules, "tau2.evaluator.evaluator_communicate", SimpleNamespace(
        CommunicateEvaluator=SimpleNamespace(calculate_reward=comm_reward),
    ))
    monkeypatch.setitem(sys.modules, "tau2.registry", SimpleNamespace(
        registry=SimpleNamespace(get_env_constructor=lambda domain: ReplayEnvironment),
    ))
    result = evaluate_tau2_prefix({"evaluation_criteria": {"reward_basis": ["DB"]}}, [])
    assert result == {"db": True, "communicate": [True]}
    assert replayed == ["valid0", "valid1"]
    assert communicated == history and communicated[2].error is True

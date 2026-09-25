import asyncio
from types import SimpleNamespace

import pytest
import tau2_agentic_rl.environment.tau2_gym as tau2_gym
from tau2_agentic_rl.environment.tau2_gym import Tau2GymAdapter


def test_database_hash_snapshot_is_instance_local() -> None:
    left = Tau2GymAdapter(task_id="0", user_model="mock", user_cache_dir="cache-a")
    right = Tau2GymAdapter(task_id="0", user_model="mock", user_cache_dir="cache-b")
    left._initial_db_hash = "left"
    right._initial_db_hash = "right"
    assert left.initial_db_hash() == "left"
    assert right.initial_db_hash() == "right"


def test_reset_rejects_upstream_max_errors_drift(monkeypatch) -> None:
    class Environment:
        def __init__(self):
            self._simulation_done = SimpleNamespace(is_set=lambda: False)
            self._orchestrator = SimpleNamespace(
                max_errors=9,
                environment=SimpleNamespace(get_db_hash=lambda: "initial"),
            )
            self._agent = SimpleNamespace(observation=[])

        def reset(self, seed=None):
            return None, {}

    monkeypatch.setattr(
        tau2_gym, "build_cached_agent_gym_env", lambda **kwargs: Environment()
    )
    adapter = Tau2GymAdapter(task_id="0", user_model="mock", max_errors=10)

    with pytest.raises(RuntimeError, match="max_errors"):
        asyncio.run(adapter.reset(seed=300))

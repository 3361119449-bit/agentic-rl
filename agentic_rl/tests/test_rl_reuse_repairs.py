import asyncio
import sys
import threading
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_fatal_rollout_channel import LocalBudget, worker_class
from test_judge_evidence import judge_inputs, judge_payload
from test_training_selection import fixture

from tau2_agentic_rl.environment.tau2_gym import Tau2GymAdapter
from tau2_agentic_rl.judge.client import DeepSeekJudge
from tau2_agentic_rl.schemas import JudgeResult
from tau2_agentic_rl.slot_recovery import interaction_retryable
from tau2_agentic_rl.training_selection import select_training_rows
from tau2_agentic_rl.user_simulation import UserSimulationRejected


@pytest.mark.parametrize("phase", ["reset", "text", "tool"])
@pytest.mark.parametrize("error", [TimeoutError("user deadline"), AttributeError("bug")])
def test_background_failure_reaches_adapter_without_losing_type(monkeypatch, phase, error):
    class FakeGym:
        def __init__(self, **kwargs):
            self._simulation_done = threading.Event()
            self._lock = threading.Lock()
            backend = SimpleNamespace(get_db_hash=lambda: "db", get_response=lambda call: None)

            def fail():
                raise error

            self._orchestrator = SimpleNamespace(run=fail, environment=backend)
            self._agent = SimpleNamespace(set_action=lambda action: self.finish(), is_agent_turn=True)

        def _log(self, *args):
            pass

        def _run_orchestrator(self):
            # Pinned Tau2 swallows the original error at this thread boundary.
            self._simulation_run = None
            try:
                self._simulation_run = self._orchestrator.run()
            except Exception:
                pass
            finally:
                self._simulation_done.set()

        def finish(self):
            thread = threading.Thread(target=self._run_orchestrator)
            thread.start()
            thread.join(timeout=2)
            assert not thread.is_alive()

        def reset(self, **kwargs):
            if phase == "reset":
                self.finish()
            return None, self._get_info()

        def step(self, action):
            self.finish()
            return None, 0, True, False, self._get_info()

        def _get_info(self):
            return {"simulation_run": getattr(self, "_simulation_run", None)}

        def _get_reward(self):
            return 0, {}

    monkeypatch.setitem(sys.modules, "tau2.gym.gym_agent", SimpleNamespace(AgentGymEnv=FakeGym))
    monkeypatch.setitem(sys.modules, "tau2.user.user_simulator", SimpleNamespace(UserSimulator=object))
    monkeypatch.setitem(sys.modules, "tau2.data_model.message", SimpleNamespace(
        AssistantMessage=lambda **kw: SimpleNamespace(**kw),
        ToolCall=lambda **kw: SimpleNamespace(**kw),
    ))
    adapter = Tau2GymAdapter(task_id="0", user_model="test")

    async def run():
        await adapter.reset()
        if phase == "text":
            await adapter.step_text("hello")
        elif phase == "tool":
            await adapter.step_tool("read", {})

    with pytest.raises(type(error)) as caught:
        asyncio.run(run())
    assert caught.value is error
    assert interaction_retryable("environment_reset", caught.value) == isinstance(error, TimeoutError)
    # A subsequent successful run must clear the old failure.
    adapter.env._simulation_done.clear()
    adapter.env._orchestrator.run = lambda: {"id": "ok"}
    adapter.env.finish()
    assert adapter.env._orchestrator_error is None
    assert adapter.env._simulation_run == {"id": "ok"}


def test_validation_user_rejection_is_not_a_job_wide_fatal_error():
    budget, queue, tags = LocalBudget(), [], []
    worker = worker_class(budget, {3: UserSimulationRejected("invalid user")}, queue, tags=tags)()
    worker.config.actor_rollout_ref.rollout.val_kwargs = SimpleNamespace(n=8)
    asyncio.run(worker._run_prompt({"uid": "val"}, {}, {"validate": True}))
    assert budget.call("fatal_rollout") is None
    assert tags[-1]["failed_session_ids"] == [3]
    assert tags[-1]["failure_kind"] == "transient_exhausted"
    assert len([item for item in queue if isinstance(item, tuple)]) == 7


@pytest.mark.parametrize("error", [RuntimeError("unknown"), AttributeError("bug"), AssertionError("alignment")])
def test_validation_deterministic_errors_remain_job_wide_fatal(error):
    budget, queue = LocalBudget(), []
    worker = worker_class(budget, {3: error}, queue)()
    worker.config.actor_rollout_ref.rollout.val_kwargs = SimpleNamespace(n=8)
    asyncio.run(worker._run_prompt({"uid": "val"}, {}, {"validate": True}))
    assert budget.call("fatal_rollout")["error_type"] == type(error).__name__


@pytest.mark.parametrize("failure", ["user", "infrastructure"])
def test_internal_validation_recovers_only_its_slot_and_counts_physical_attempts(scratch_dir, monkeypatch, failure):
    from test_rollout_integration import minimal_loop
    from test_slot_recovery import configure

    from tau2_agentic_rl.slot_recovery import (
        RolloutInfrastructureError,
        SlotRecoveryExhausted,
    )

    monkeypatch.delenv("EVALUATION_MANIFEST_ID", raising=False)
    loop, _ = minimal_loop(scratch_dir)
    configure(loop)
    loop.project["user_sim_filter"] = {"enabled": True, "max_resamples": 2}
    attempts = []
    before = loop.shared_budget.call("rollout_counts", "val")
    train_before = loop.shared_budget.call("rollout_counts", "train")

    async def fail(params, **kwargs):
        attempts.append(kwargs)
        if failure == "user":
            raise UserSimulationRejected("invalid user")
        raise RolloutInfrastructureError("environment_reset", kwargs["trajectory_id"]) from TimeoutError("user timeout")

    loop._run_trajectory = fail
    with pytest.raises(SlotRecoveryExhausted):
        asyncio.run(loop._run_valid_trajectory({}, uid="val", session_id=3,
            extra_info={"task_id": "11", "split": "internal_dev", "environment_seed": 42}))
    assert len(attempts) == 3
    assert all(attempt["uid"] == "val" and attempt["session_id"] == 3 for attempt in attempts)
    counts = loop.shared_budget.call("rollout_counts", "val")
    counts = {key: value - before[key] for key, value in counts.items()}
    assert counts == {"initial": 1, "infrastructure": 2 if failure == "infrastructure" else 0,
                      "user": 2 if failure == "user" else 0}
    assert loop.shared_budget.call("rollout_counts", "train") == train_before


def test_judge_accepts_permutations_and_canonicalizes_all_sections():
    inputs, payload = judge_inputs(), judge_payload()
    for field in ("semantic_checks", "transfer_semantic_checks", "mandatory_policy_checks"):
        definitions = inputs["transfer_rule"]["semantic_checks"] if field == "transfer_semantic_checks" else inputs[field]
        definitions.append({"criterion_id": definitions[0]["criterion_id"] + "2"})
        second = deepcopy(payload[field][0])
        second["criterion_id"] += "2"
        payload[field] = [second, payload[field][0]]
    result = JudgeResult.model_validate(payload)
    DeepSeekJudge._validate_requested_criteria(result, inputs)
    for field in ("semantic_checks", "transfer_semantic_checks", "mandatory_policy_checks"):
        assert getattr(result, field)[1].criterion_id.endswith("2")


@pytest.mark.parametrize("field", ["semantic_checks", "transfer_semantic_checks", "mandatory_policy_checks"])
@pytest.mark.parametrize("change", ["duplicate", "missing", "extra", "rubric_duplicate"])
def test_judge_still_rejects_invalid_membership(field, change):
    inputs, payload = judge_inputs(), judge_payload()
    definitions = inputs["transfer_rule"]["semantic_checks"] if field == "transfer_semantic_checks" else inputs[field]
    if change == "duplicate":
        payload[field].append(deepcopy(payload[field][0]))
    elif change == "missing":
        payload[field] = []
    elif change == "extra":
        payload[field][0]["criterion_id"] = "invented"
    else:
        definitions.append(deepcopy(definitions[0]))
    with pytest.raises(ValueError):
        DeepSeekJudge._validate_requested_criteria(JudgeResult.model_validate(payload), inputs)


def test_v4_keeps_refusal_and_policy_tasks_without_initial_progress():
    kwargs = fixture()
    kwargs["tasks"]["0"]["evaluation_criteria"] = {"nl_assertions": ["Agent should refuse cancellation"]}
    rows, manifest = select_training_rows(**kwargs, mode="task_or_local_credit")
    assert [row["extra_info"]["task_id"] for row in rows] == ["0", "1", "2", "3"]
    assert manifest["mode"] == "task_or_local_credit"
    assert manifest["tasks"][0]["reason"] == "task_or_local_credit"
    assert not any(check["included"] for check in manifest["tasks"][0]["checks"])
    kwargs["rows"][0]["extra_info"]["split"] = "internal_dev"
    with pytest.raises(ValueError):
        select_training_rows(**kwargs, mode="task_or_local_credit")

import asyncio
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_procredit_runtime import production_method
from test_rollout_integration import minimal_loop

from tau2_agentic_rl.advantages import CreditConfig
from tau2_agentic_rl.slot_recovery import (
    RolloutInfrastructureError,
    SlotRecoveryExhausted,
)


def configure(loop):
    loop.project["credit"] = {"version": "procredit-turn-v4"}
    loop.project["slot_recovery"] = {
        "max_resamples": 2, "max_scoring_retries": 2, "on_exhaustion": "stop",
    }
    loop.project["user_sim_filter"] = {"enabled": False}


def test_eight_slot_group_only_resamples_the_failed_slot(scratch_dir, monkeypatch):
    monkeypatch.delenv("EVALUATION_MANIFEST_ID", raising=False)
    calls, captured, outputs = Counter(), [], {}

    async def scenario():
        loops = [minimal_loop(scratch_dir / str(i))[0] for i in range(8)]
        for loop in loops:
            configure(loop)

        async def generate(params, **kwargs):
            slot = kwargs["extra_info"]["slot"]
            calls[slot] += 1
            captured.append(deepcopy(kwargs))
            if slot == 3 and calls[slot] == 1:
                raise RolloutInfrastructureError("model_generation", kwargs["trajectory_id"])
            result = SimpleNamespace(
                response_ids=[slot, 9], response_logprobs=[-0.3, -0.4],
                extra_fields={"policy_valid": False if slot == 4 else True},
            )
            outputs[slot] = result
            return result

        for loop in loops:
            loop._run_trajectory = generate
        return await asyncio.gather(*[
            loop._run_valid_trajectory({}, trajectory_id=f"{i:032x}",
                extra_info={"slot": i, "task_id": "11", "split": "train", "environment_seed": 42},
                global_steps=7, lease="same-policy-lease")
            for i, loop in enumerate(loops)
        ])

    results = asyncio.run(scenario())
    assert calls == Counter({**dict.fromkeys(range(8), 1), 3: 2})
    assert all(results[i] is outputs[i] for i in range(8))
    failed = [item for item in captured if item["extra_info"]["slot"] == 3]
    assert failed[0]["trajectory_id"] != failed[1]["trajectory_id"]
    assert all(item["global_steps"] == 7 and item["lease"] == "same-policy-lease" for item in failed)
    assert all(item["extra_info"]["environment_seed"] == 42 for item in failed)
    assert calls[4] == 1  # A policy violation is training data, never an infra retry.


def test_exhausted_slot_stops_without_infinite_reroll(scratch_dir, monkeypatch):
    monkeypatch.delenv("EVALUATION_MANIFEST_ID", raising=False)
    loop, _ = minimal_loop(scratch_dir)
    configure(loop)
    calls = []

    async def broken(params, **kwargs):
        calls.append(kwargs["trajectory_id"])
        raise RolloutInfrastructureError("environment_reset", kwargs["trajectory_id"])

    loop._run_trajectory = broken
    with pytest.raises(SlotRecoveryExhausted):
        asyncio.run(loop._run_valid_trajectory({}, extra_info={"task_id": "11", "environment_seed": 42}))
    assert len(calls) == len(set(calls)) == 3


def test_v4_buffer_never_evicts_seven_successes_due_to_one_exhausted_slot():
    class Parent:
        failure_keys = {"train": {"group"}}
        def _terminal_eviction_reasons(self, step, partition):
            return set(), set(), set(), {}

    cls = production_method(
        "CappedDynamicReplayBuffer", "_terminal_eviction_reasons", Parent,
        SlotRecoveryExhausted=SlotRecoveryExhausted,
    )
    buffer = cls()
    buffer.credit_config = CreditConfig(version="procredit-turn-v4")
    with pytest.raises(SlotRecoveryExhausted):
        buffer._terminal_eviction_reasons(1, "train")
    assert buffer.failure_keys["train"] == {"group"}


@pytest.mark.parametrize("error", [asyncio.CancelledError, ValueError])
def test_slot_does_not_swallow_cancellation_or_unclassified_errors(scratch_dir, error):
    loop, _ = minimal_loop(scratch_dir)
    configure(loop)
    calls = []

    async def broken(params, **kwargs):
        calls.append(kwargs)
        raise error()

    loop._run_trajectory = broken
    with pytest.raises(error):
        asyncio.run(loop._run_valid_trajectory({}, extra_info={"task_id": "11"}))
    assert len(calls) == 1


def test_user_and_infrastructure_replacements_share_slot_but_have_separate_budgets(scratch_dir):
    from tau2_agentic_rl.user_simulation import UserSimulationRejected

    loop, _ = minimal_loop(scratch_dir)
    configure(loop)
    loop.project["user_sim_filter"] = {"enabled": True, "max_resamples": 1}
    captured = []

    async def generate(params, **kwargs):
        captured.append(deepcopy(kwargs))
        if len(captured) == 1:
            raise UserSimulationRejected("invalid user")
        if len(captured) == 2:
            raise RolloutInfrastructureError("environment_reset", kwargs["trajectory_id"])
        return SimpleNamespace(extra_fields={})

    loop._run_trajectory = generate
    result = asyncio.run(loop._run_valid_trajectory(
        {}, extra_info={"task_id": "11", "environment_seed": 42},
    ))
    seeds = [item["extra_info"]["environment_seed"] for item in captured]
    assert seeds[0] != seeds[1] == seeds[2]
    assert result.extra_fields["slot_recovery"]["infrastructure_replacements"] == 1
    assert result.extra_fields["slot_recovery"]["user_replacements"] == 1


def test_evaluation_infrastructure_failure_is_left_to_evaluation_driver(scratch_dir, monkeypatch):
    monkeypatch.setenv("EVALUATION_MANIFEST_ID", "evaluation")
    loop, _ = minimal_loop(scratch_dir)
    configure(loop)
    calls = []

    async def broken(params, **kwargs):
        calls.append(kwargs)
        raise RolloutInfrastructureError("environment_reset", "frozen")

    loop._run_trajectory = broken
    with pytest.raises(RolloutInfrastructureError):
        asyncio.run(loop._run_valid_trajectory({}, extra_info={"task_id": "11"}))
    assert len(calls) == 1


@pytest.mark.parametrize("phase", [
    "policy_version_alignment", "rollout_log_probs", "token_alignment",
    "prompt_initialization", "tool_result_alignment", "unrecognized_phase",
])
def test_consistency_errors_stop_without_replacement(scratch_dir, phase):
    loop, _ = minimal_loop(scratch_dir)
    configure(loop)
    calls = []

    async def broken(params, **kwargs):
        calls.append(kwargs)
        raise RolloutInfrastructureError(phase, kwargs["trajectory_id"])

    loop._run_trajectory = broken
    with pytest.raises(RolloutInfrastructureError):
        asyncio.run(loop._run_valid_trajectory({}, extra_info={"task_id": "11"}))
    assert len(calls) == 1


def test_configuration_error_wrapped_in_service_phase_is_not_retried(scratch_dir):
    loop, _ = minimal_loop(scratch_dir)
    configure(loop)
    calls = []

    async def broken(params, **kwargs):
        calls.append(kwargs)
        raise RolloutInfrastructureError("environment_reset", kwargs["trajectory_id"]) from ValueError("bad config")

    loop._run_trajectory = broken
    with pytest.raises(RolloutInfrastructureError):
        asyncio.run(loop._run_valid_trajectory({}, extra_info={"task_id": "11"}))
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ["cause", "context", "group"])
def test_nested_contract_errors_cannot_be_hidden_by_service_wrappers(scratch_dir, kind):
    loop, _ = minimal_loop(scratch_dir)
    configure(loop)
    calls = []

    async def broken(params, **kwargs):
        calls.append(kwargs)
        try:
            if kind == "group":
                raise ExceptionGroup("backend", [ConnectionError("service"), ValueError("bad config")])
            try:
                raise ValueError("bad config")
            except ValueError as exc:
                if kind == "cause":
                    raise RuntimeError("adapter") from exc
                raise RuntimeError("adapter")
        except Exception as exc:
            raise RolloutInfrastructureError("environment_reset", kwargs["trajectory_id"]) from exc

    loop._run_trajectory = broken
    with pytest.raises(RolloutInfrastructureError):
        asyncio.run(loop._run_valid_trajectory({}, extra_info={"task_id": "11"}))
    assert len(calls) == 1


def test_service_exception_group_still_replaces_only_its_slot(scratch_dir):
    loop, _ = minimal_loop(scratch_dir)
    configure(loop)
    calls = []

    async def generate(params, **kwargs):
        calls.append(kwargs["trajectory_id"])
        if len(calls) == 1:
            raise RolloutInfrastructureError("environment_reset", kwargs["trajectory_id"]) from ExceptionGroup(
                "temporary service failures", [ConnectionError("offline"), TimeoutError("timeout")],
            )
        return SimpleNamespace(extra_fields={})

    loop._run_trajectory = generate
    output = asyncio.run(loop._run_valid_trajectory({}, extra_info={"task_id": "11"}))
    assert len(set(calls)) == 2
    assert output.extra_fields["slot_recovery"]["infrastructure_replacements"] == 1


@pytest.mark.parametrize("failure", ["infrastructure", "invalid_user"])
@pytest.mark.parametrize("handler_pending", [False, True])
def test_outer_run_cancellation_drains_current_call_without_starting_a_replacement(scratch_dir, failure, handler_pending):
    from tau2_agentic_rl.concurrency import SharedBudget
    from tau2_agentic_rl.user_simulation import UserSimulationRejected

    loop, scope = minimal_loop(scratch_dir)
    configure(loop)
    loop.project["user_sim_filter"] = {"enabled": True, "max_resamples": 2}
    budget = SharedBudget({"trajectories": 1, "user_api": 1, "judge_api": 1})
    scope["SharedBudget"] = lambda *args, **kwargs: budget
    calls = []

    async def scenario():
        started, finish = asyncio.Event(), asyncio.Event()

        async def generate(params, **kwargs):
            calls.append(kwargs["trajectory_id"])
            started.set()
            await finish.wait()  # A real in-flight call must drain under its lease.
            if failure == "invalid_user":
                raise UserSimulationRejected("bad user")
            raise RolloutInfrastructureError("environment_reset", kwargs["trajectory_id"])

        loop._run_trajectory = generate
        task = asyncio.create_task(loop.run({}, extra_info={"task_id": "11"}))
        await started.wait()
        try:
            if handler_pending:
                finish.set()  # Inner failure is ready before the parent cancel handler.
            task.cancel()
            if not handler_pending:
                await asyncio.sleep(0)
            assert budget.call("snapshot")["active"]["trajectories"] == 1
            assert not task.done()
        finally:
            finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert budget.call("snapshot")["active"]["trajectories"] == 0

    asyncio.run(scenario())
    assert len(calls) == 1


@pytest.mark.parametrize("metadata", [
    {"failure_phase": "token_alignment"},
    {"failure_phase": "environment_reset", "interaction_retryable": False},
])
def test_evaluation_does_not_turn_contract_failure_into_missing_slot(scratch_dir, metadata):
    import json

    from tau2_agentic_rl.evaluation import evaluation_coverage, initialize_evaluation

    manifest = initialize_evaluation(scratch_dir / "eval", {
        "task_ids": ["11"], "samples_per_task": 4, "split": "internal_dev",
        "record_split": "internal_dev", "reward_judge_enabled": False,
    }, resume=False)
    records = scratch_dir / "records"
    records.mkdir()
    row = {
        "trajectory_id": "bad", "task_id": "11", "split": "internal_dev",
        "termination_reason": "infrastructure_error", "official_scores": None,
        "metadata": {
            "evaluation_manifest_id": manifest["manifest_id"], "evaluation_sample_index": 0,
            **metadata,
        },
    }
    (records / "bad.json").write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="non-retryable"):
        evaluation_coverage(records, manifest)

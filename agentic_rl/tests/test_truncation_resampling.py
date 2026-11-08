"""Truncation replacements are bounded per training slot, distinct from service retry."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tau2_agentic_rl.slot_recovery import run_training_slot

PROJECT = {"slot_recovery": {"max_resamples": 2, "max_truncation_resamples": 2},
           "user_sim_filter": {"enabled": False}}


@pytest.mark.parametrize("reasons,expected", [
    (["hard_turn_limit", "user_stop"], 2),
    (["budget_exhausted", "context_window_exceeded", "user_stop"], 3),
    (["hard_turn_limit"] * 5, 3),
])
def test_only_truncated_slot_is_replaced_at_most_twice(reasons, expected):
    calls = []
    kinds = []

    async def generate(params, **kwargs):
        calls.append(deepcopy(kwargs))
        return SimpleNamespace(extra_fields={
            "termination_reason": reasons[min(len(calls) - 1, len(reasons) - 1)]})

    async def attempt(kind):
        kinds.append(kind)

    result = asyncio.run(run_training_slot(generate, {}, project=PROJECT,
        kwargs={"trajectory_id": "a" * 32, "extra_info": {
            "split": "train", "task_id": "18", "slot": 3, "environment_seed": 42}},
        on_attempt=attempt))
    assert len(calls) == expected
    assert len({c["trajectory_id"] for c in calls}) == expected
    assert all(c["extra_info"]["slot"] == 3 for c in calls)
    assert kinds == ["initial"] + ["truncation"] * (expected - 1)
    assert result.extra_fields["slot_recovery"]["truncation_replacements"] == expected - 1
    assert len(result.extra_fields["slot_recovery"]["truncated_attempts"]) == (
        expected if reasons[-1] != "user_stop" else expected - 1)


@pytest.mark.parametrize("split", ["internal_dev", "test", "val"])
def test_validation_never_resamples_truncation(split):
    calls = []

    async def generate(params, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(extra_fields={"termination_reason": "hard_turn_limit"})

    asyncio.run(run_training_slot(generate, {}, project=PROJECT,
        kwargs={"trajectory_id": "a" * 32, "extra_info": {"split": split}}))
    assert len(calls) == 1


def test_legacy_slot_configuration_does_not_gain_truncation_retries():
    calls = []

    async def generate(params, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(extra_fields={"termination_reason": "budget_exhausted"})

    asyncio.run(run_training_slot(generate, {},
        project={"slot_recovery": {"max_resamples": 2}, "user_sim_filter": {"enabled": False}},
        kwargs={"trajectory_id": "a" * 32}))
    assert len(calls) == 1


def test_truncation_retries_are_visible_in_shared_budget():
    from tau2_agentic_rl.concurrency import SharedBudget
    budget = SharedBudget({"trajectories": 1, "user_api": 1, "judge_api": 1})
    budget.call("record_rollout_attempt", "train", "initial")
    budget.call("record_rollout_attempt", "train", "truncation")
    budget.call("record_rollout_attempt", "train", "truncation")
    assert budget.call("rollout_counts", "train")["truncation"] == 2


def test_generic_grpo_training_also_resamples_only_truncated_slot(scratch_dir, monkeypatch):
    from test_rollout_integration import minimal_loop
    monkeypatch.delenv("EVALUATION_MANIFEST_ID", raising=False)
    loop, _ = minimal_loop(scratch_dir)
    loop.project.update(deepcopy(PROJECT))
    loop.project["reward"] = {"truncation_policy": "zero-trajectory-v1",
                              "truncation_multiplier": 0}
    calls = []

    async def generate(params, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(extra_fields={
            "termination_reason": "hard_turn_limit" if len(calls) < 3 else "user_stop"})

    async def snapshot(*args):
        return {}

    loop.shared_budget = SimpleNamespace(acall=snapshot)
    loop._run_trajectory = generate
    output = asyncio.run(loop._run_valid_trajectory({}, trajectory_id="a" * 32,
                          extra_info={"split": "train", "task_id": "18"}))
    assert len(calls) == 3
    assert output.extra_fields["slot_recovery"]["truncation_replacements"] == 2


def test_truncation_and_infrastructure_budgets_do_not_reset_each_other():
    from tau2_agentic_rl.slot_recovery import RolloutInfrastructureError
    calls = []

    async def generate(params, **kwargs):
        calls.append(kwargs)
        if len(calls) in [2, 4]:
            raise RolloutInfrastructureError("model_generation", kwargs["trajectory_id"]) from ConnectionError("fixture")
        return SimpleNamespace(extra_fields={"termination_reason": "hard_turn_limit"})

    result = asyncio.run(run_training_slot(generate, {}, project=PROJECT,
                           kwargs={"trajectory_id": "a" * 32}))
    assert len(calls) == 5
    assert result.extra_fields["slot_recovery"]["truncation_replacements"] == 2
    assert result.extra_fields["slot_recovery"]["infrastructure_replacements"] == 2
    assert len(result.extra_fields["slot_recovery"]["truncated_attempts"]) == 3


def test_cancel_after_truncated_attempt_never_starts_replacement():
    calls = []

    async def generate(params, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(extra_fields={"termination_reason": "hard_turn_limit"})

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_training_slot(generate, {}, project=PROJECT,
            kwargs={"trajectory_id": "a" * 32}, stop_requested=lambda: bool(calls)))
    assert len(calls) == 1

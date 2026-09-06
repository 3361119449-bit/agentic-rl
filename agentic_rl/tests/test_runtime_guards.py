"""Queue clocks are simulated; these tests do not wait 30 minutes or use APIs."""

import asyncio
from types import SimpleNamespace

import pytest

from tau2_agentic_rl import concurrency
from tau2_agentic_rl.budget import ContextBudget
from tau2_agentic_rl.concurrency import (
    QueueWaitError,
    SharedBudget,
    queue_options_from_project,
)
from tau2_agentic_rl.initial_prompt import (
    inspect_initial_prompt,
    require_initial_prompt_fits,
    summarize_initial_prompts,
)


@pytest.mark.parametrize("sync", [True, False])
@pytest.mark.parametrize("kind", ["progress", "stalled", "deadline"])
def test_queue_distinguishes_slow_progress_from_stall_and_explicit_deadline(
    monkeypatch, sync, kind
):
    budget = SharedBudget(
        {"trajectories": 1, "user_api": 1, "judge_api": 1},
        queue_timeout_seconds=1200 if kind == "deadline" else None,
    )
    clock = [0.0]
    # Replace only the module's clock, not asyncio's own event-loop clock.
    monkeypatch.setattr(
        concurrency, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    holder = "held-by-running-environment"
    assert budget.call("try_acquire", "trajectories", holder, budget.limits)

    def tick(_):
        clock[0] += 600
        if kind != "stalled":
            budget.call("heartbeat", "trajectories", holder)
        if kind == "progress" and clock[0] >= 3600:
            budget.call("release", "trajectories", holder)

    async def async_tick(delay):
        tick(delay)

    concurrency.time.sleep = tick
    monkeypatch.setattr(concurrency, "asyncio", SimpleNamespace(sleep=async_tick))

    def enter_sync():
        with budget.slot("trajectories") as lease:
            assert kind == "progress"
            assert lease.queue_wait_seconds == 3600

    async def enter_async():
        async with budget.aslot("trajectories") as lease:
            assert kind == "progress"
            assert lease.queue_wait_seconds == 3600

    try:
        if kind == "progress":
            enter_sync() if sync else asyncio.run(enter_async())
            assert budget.call("snapshot")["active"]["trajectories"] == 0
        else:
            with pytest.raises(QueueWaitError) as error:
                enter_sync() if sync else asyncio.run(enter_async())
            assert error.value.details["reason"] == (
                "stalled" if kind == "stalled" else "deadline_exceeded"
            )
            # Timed-out waiter must not free an executing environment's lease.
            assert budget.call("snapshot")["active"]["trajectories"] == 1
    finally:
        if kind != "progress":
            budget.call("release", "trajectories", holder)


@pytest.mark.parametrize(
    "key,value",
    [
        ("queue_timeout_seconds", -1),
        ("queue_timeout_seconds", True),
        ("queue_timeout_seconds", float("nan")),
        ("queue_stall_timeout_seconds", None),
        ("queue_stall_timeout_seconds", 0),
    ],
)
def test_queue_options_reject_unbounded_stall_or_invalid_time(key, value):
    with pytest.raises(ValueError, match=key):
        queue_options_from_project({"rollout": {key: value}})


@pytest.mark.parametrize(
    "count,limit,expected",
    [
        (8192, 8192, []),
        (8193, 8192, ["initial_prompt_limit_exceeded"]),
        (15168, 16000, []),
        (15169, 16000, ["total_context_reserve_exceeded"]),
        (
            17000,
            8192,
            ["initial_prompt_limit_exceeded", "total_context_reserve_exceeded"],
        ),
    ],
)
def test_initial_prompt_boundaries_include_all_reserves(count, limit, expected):
    measurement = inspect_initial_prompt("task-7", count, limit, ContextBudget())
    assert measurement["violations"] == expected
    assert measurement["prompt_plus_reserve_tokens"] == count + 1216
    if expected:
        with pytest.raises(ValueError, match=f"task-7.*{count}"):
            require_initial_prompt_fits(measurement)
    else:
        require_initial_prompt_fits(measurement)


def test_prompt_summary_keeps_missing_and_reset_failures_visible():
    rows = [
        inspect_initial_prompt(str(i), 100 * (i + 1), 8192, ContextBudget())
        for i in range(20)
    ]
    report = summarize_initial_prompts(rows, [str(i) for i in range(20)])
    assert report["tokens"] == {
        "min": 100,
        "max": 2000,
        "p50": 1000,
        "p90": 1800,
        "p95": 1900,
        "p99": 2000,
    }
    assert report["all_requested_samples_passed"]
    rows.append({"task_id": "failed", "status": "reset_error"})
    report = summarize_initial_prompts(rows, ["failed", "missing"])
    assert report["missing_task_ids"] == ["failed", "missing"]
    assert not report["all_requested_samples_passed"]


def test_preflight_uses_same_real_reset_inputs_and_cleans_up(monkeypatch, scratch_dir):
    from scripts import check_initial_prompts as script
    from tau2_agentic_rl.config import load_yaml
    from tau2_agentic_rl.tooling import CONFIRMATION_PROTOCOL

    calls = []
    tools = [{"type": "function", "function": {"name": "lookup"}}]

    class Environment:
        policy, tool_schemas = "actual policy", tools

        def __init__(self, **kwargs):
            self.task_id = kwargs["task_id"]

        async def reset(self, seed):
            calls.append((self.task_id, seed))
            if self.task_id == "4":
                raise RuntimeError("reset unavailable")
            return [{"role": "user", "content": "actual initial user"}]

        async def force_cleanup_stop(self):
            calls.append((self.task_id, "cleanup"))

    def encode(messages, **kwargs):
        assert messages == [
            {"role": "system", "content": f"actual policy\n\n{CONFIRMATION_PROTOCOL}"},
            {"role": "user", "content": "actual initial user"},
        ]
        assert kwargs["tools"] == tools and kwargs["truncation"] is False
        return [1] * 8193

    monkeypatch.setattr(script, "Tau2GymAdapter", Environment)
    project = load_yaml(script.ROOT / "configs/rl/airline_grpo_v1.yaml")
    rows = asyncio.run(
        script.measure_live(
            SimpleNamespace(cache_dir=scratch_dir),
            ["0", "4"],
            project,
            SimpleNamespace(apply_chat_template=encode),
        )
    )
    assert rows[0]["initial_prompt_tokens"] == 8193 and rows[0]["status"] == "rejected"
    assert rows[1]["status"] == "initialization_error"
    assert calls == [("0", 0), ("0", "cleanup"), ("4", 1), ("4", "cleanup")]


def test_offline_preflight_does_not_guess_unmeasured_old_record_lengths(scratch_dir):
    import json

    from scripts.check_initial_prompts import recorded_measurements

    (scratch_dir / "old.json").write_text(
        json.dumps(
            {
                "task_id": "0",
                "trajectory_id": "old",
                "trajectory_tokens": 10000,
                "messages": [
                    {"role": "system", "content": "history without tool schemas"}
                ],
            }
        ),
        encoding="utf-8",
    )
    rows = recorded_measurements(scratch_dir, ["0"])
    assert rows == [
        {
            "task_id": "0",
            "trajectory_id": "old",
            "status": "not_measured",
            "failure_phase": None,
        }
    ]

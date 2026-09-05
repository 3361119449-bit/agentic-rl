"""One shared budget per Ray job, not one semaphore per AgentLoop instance."""

import asyncio
import os
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from uuid import uuid4

from tau2_agentic_rl.config import load_runtime_config


def limits_from_project(project):
    rollout = project["rollout"]
    keys = {
        "trajectories": "max_active_trajectories",
        "user_api": "user_api_max_inflight",
        "judge_api": "judge_api_max_inflight",
    }
    limits = {name: rollout[key] for name, key in keys.items()}
    if any(type(value) is not int or value < 1 for value in limits.values()):
        raise ValueError("global concurrency limits must be positive integers")
    return limits


class BudgetState:
    """Fail-closed lease bookkeeping, also directly testable without Ray."""

    def __init__(self, limits):
        self.limits = dict(limits)
        self.active = {key: set() for key in limits}
        self.peak = dict.fromkeys(limits, 0)
        self.lock = threading.Lock()

    def try_acquire(self, resource, lease, limits):
        with self.lock:
            if limits != self.limits:
                raise ValueError("workers disagree on the shared concurrency limits")
            active = self.active[resource]
            if lease in active:
                raise ValueError("duplicate concurrency lease")
            if len(active) >= self.limits[resource]:
                return False
            active.add(lease)
            self.peak[resource] = max(self.peak[resource], len(active))
            return True

    def release(self, resource, lease):
        with self.lock:
            self.active[resource].remove(lease)

    def snapshot(self):
        with self.lock:
            return {
                "limits": self.limits,
                "active": {key: len(value) for key, value in self.active.items()},
                "peak_active_trajectories": self.peak["trajectories"],
                "peak_user_api_inflight": self.peak["user_api"],
                "peak_judge_api_inflight": self.peak["judge_api"],
            }


_local_budgets = {}
_local_lock = threading.Lock()


class SharedBudget:
    def __init__(self, limits, *, require_ray=False):
        self.limits = limits
        try:
            import ray
        except ImportError:
            ray = None
        self.ray = ray if ray is not None and ray.is_initialized() else None
        if require_ray and self.ray is None:
            raise RuntimeError("rollout requires a Ray-job-wide concurrency budget")
        if self.ray is not None:
            # All workers in this job race safely to obtain the SAME actor.
            # No detached lifetime: the driver owns the experiment's lifetime.
            name = f"tau2-budget-{ray.get_runtime_context().get_job_id()}"
            self.state = (
                ray.remote(num_cpus=0)(BudgetState)
                .options(
                    name=name,
                    get_if_exists=True,
                    max_restarts=0,
                )
                .remote(limits)
            )
        else:
            # Offline rejudging is single-process; its thread/async clients share
            # this state. Live multi-worker rollout may not use this fallback.
            key = tuple(sorted(limits.items()))
            with _local_lock:
                self.state = _local_budgets.setdefault(key, BudgetState(limits))

    def call(self, method, *args):
        fn = getattr(self.state, method)
        return self.ray.get(fn.remote(*args)) if self.ray else fn(*args)

    async def acall(self, method, *args):
        fn = getattr(self.state, method)
        return await fn.remote(*args) if self.ray else fn(*args)

    @contextmanager
    def slot(self, resource, timeout=1800):
        lease, start = uuid4().hex, time.monotonic()
        while not self.call("try_acquire", resource, lease, self.limits):
            if time.monotonic() - start > timeout:
                raise TimeoutError(
                    "shared concurrency lease timed out; possible failed worker"
                )
            time.sleep(0.05)
        try:
            yield
        finally:
            self.call("release", resource, lease)

    @asynccontextmanager
    async def aslot(self, resource, timeout=1800):
        lease, start = uuid4().hex, time.monotonic()
        # Acquire is intentionally synchronous/short: cancellation must not lose
        # a lease in flight between the actor and the caller. There is no queued
        # acquire on the actor. Waits on occupied slots are async/cancellable.
        while not self.call("try_acquire", resource, lease, self.limits):
            if time.monotonic() - start > timeout:
                raise TimeoutError(
                    "shared concurrency lease timed out; possible failed worker"
                )
            await asyncio.sleep(0.05)
        try:
            yield
        finally:
            self.call("release", resource, lease)


def api_budget():
    path = os.environ.get("AGENTIC_RL_CONFIG")
    limits = (
        limits_from_project(load_runtime_config(path))
        if path
        else {
            "trajectories": 1,
            "user_api": 1,
            "judge_api": 1,
        }
    )
    return SharedBudget(limits)

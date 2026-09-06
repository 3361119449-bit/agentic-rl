"""One shared budget per Ray job, not one semaphore per AgentLoop instance."""

import asyncio
import math
import os
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
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


def queue_options_from_project(project):
    rollout = project["rollout"]
    options = {
        "queue_timeout_seconds": rollout.get("queue_timeout_seconds"),
        "queue_stall_timeout_seconds": rollout.get("queue_stall_timeout_seconds", 1800),
    }
    for key, value in options.items():
        if value is None and key == "queue_timeout_seconds":
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"{key} must be positive and finite (only queue timeout may be null)"
            )
    return options


class QueueWaitError(TimeoutError):
    """A waiting caller failed, not an executing trajectory or API request."""

    def __init__(self, resource, kind, waited, stalled):
        self.details = {
            "resource": resource,
            "reason": kind,
            "queue_wait_seconds": waited,
            "no_progress_seconds": stalled,
        }
        super().__init__(
            f"{resource} queue {kind}: waited {waited:.1f}s, no shared progress "
            f"for {stalled:.1f}s; no active lease was reclaimed"
        )


class _QueueWait:
    def __init__(self, resource, revision, timeout, stall_timeout):
        self.resource, self.revision = resource, revision
        self.timeout, self.stall_timeout = timeout, stall_timeout
        self.start = self.last_progress = time.monotonic()

    def check(self, revision):
        now = time.monotonic()
        if revision != self.revision:
            self.revision, self.last_progress = revision, now
        waited, stalled = now - self.start, now - self.last_progress
        if self.timeout is not None and waited >= self.timeout:
            raise QueueWaitError(self.resource, "deadline_exceeded", waited, stalled)
        if stalled >= self.stall_timeout:
            raise QueueWaitError(self.resource, "stalled", waited, stalled)


@dataclass(frozen=True)
class BudgetLease:
    budget: "SharedBudget"
    resource: str
    lease_id: str
    queue_wait_seconds: float

    def progress(self):
        """Call after REAL work (reset, generation, step), not on a timer."""
        self.budget.call("heartbeat", self.resource, self.lease_id)


class BudgetState:
    """Fail-closed lease bookkeeping, also directly testable without Ray."""

    def __init__(self, limits):
        self.limits = dict(limits)
        self.active = {key: set() for key in limits}
        self.peak = dict.fromkeys(limits, 0)
        self.progress_revision = dict.fromkeys(limits, 0)
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
            self.progress_revision[resource] += 1
            self.peak[resource] = max(self.peak[resource], len(active))
            return True

    def release(self, resource, lease):
        with self.lock:
            self.active[resource].remove(lease)
            self.progress_revision[resource] += 1

    def heartbeat(self, resource, lease):
        with self.lock:
            if lease not in self.active[resource]:
                raise ValueError("cannot report progress without an active lease")
            self.progress_revision[resource] += 1

    def progress(self, resource):
        with self.lock:
            return self.progress_revision[resource]

    def snapshot(self):
        with self.lock:
            return {
                "limits": self.limits,
                "active": {key: len(value) for key, value in self.active.items()},
                "progress_revision": dict(self.progress_revision),
                "peak_active_trajectories": self.peak["trajectories"],
                "peak_user_api_inflight": self.peak["user_api"],
                "peak_judge_api_inflight": self.peak["judge_api"],
            }


_local_budgets = {}
_local_lock = threading.Lock()


class SharedBudget:
    def __init__(
        self,
        limits,
        *,
        require_ray=False,
        queue_timeout_seconds=None,
        queue_stall_timeout_seconds=1800,
    ):
        self.limits = limits
        options = queue_options_from_project(
            {
                "rollout": {
                    "queue_timeout_seconds": queue_timeout_seconds,
                    "queue_stall_timeout_seconds": queue_stall_timeout_seconds,
                }
            }
        )
        self.queue_timeout = options["queue_timeout_seconds"]
        self.stall_timeout = options["queue_stall_timeout_seconds"]
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
    def slot(self, resource):
        lease = uuid4().hex
        wait = _QueueWait(
            resource,
            self.call("progress", resource),
            self.queue_timeout,
            self.stall_timeout,
        )
        while not self.call("try_acquire", resource, lease, self.limits):
            wait.check(self.call("progress", resource))
            time.sleep(0.05)
        try:
            yield BudgetLease(self, resource, lease, time.monotonic() - wait.start)
        finally:
            self.call("release", resource, lease)

    @asynccontextmanager
    async def aslot(self, resource):
        lease = uuid4().hex
        wait = _QueueWait(
            resource,
            self.call("progress", resource),
            self.queue_timeout,
            self.stall_timeout,
        )
        # Acquire is intentionally synchronous/short: cancellation must not lose
        # a lease in flight between the actor and the caller. There is no queued
        # acquire on the actor. Waits on occupied slots are async/cancellable.
        while not self.call("try_acquire", resource, lease, self.limits):
            wait.check(self.call("progress", resource))
            await asyncio.sleep(0.05)
        try:
            yield BudgetLease(self, resource, lease, time.monotonic() - wait.start)
        finally:
            self.call("release", resource, lease)


def api_budget():
    path = os.environ.get("AGENTIC_RL_CONFIG")
    project = load_runtime_config(path) if path else {"rollout": {}}
    limits = (
        limits_from_project(project)
        if path
        else {
            "trajectories": 1,
            "user_api": 1,
            "judge_api": 1,
        }
    )
    return SharedBudget(limits, **queue_options_from_project(project))

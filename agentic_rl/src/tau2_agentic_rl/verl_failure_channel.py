"""Preserve failure severity before pinned veRL reduces it to status=failure."""

import asyncio
import logging

import ray
import transfer_queue as tq
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ, AgentLoopWorkerTQ

from tau2_agentic_rl.concurrency import QueueWaitError, api_budget
from tau2_agentic_rl.failures import raise_if_fatal
from tau2_agentic_rl.slot_recovery import (
    RolloutInfrastructureError,
    SlotRecoveryExhausted,
    interaction_retryable,
)
from tau2_agentic_rl.user_simulation import UserSimulationRejected

logger = logging.getLogger(__name__)
# The pinned TQ worker is already a Ray ActorClass. Its underlying Python class
# keeps the exact veRL generation and postprocessing APIs while adding a latch.
_TQWorker = AgentLoopWorkerTQ.__ray_metadata__.modified_class


@ray.remote
class FailureAwareAgentLoopWorkerTQ(_TQWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failure_budget = api_budget()

    def _report_failure(self, error, *, uid, partition_id, session_id=None):
        transient = (
            partition_id == "val" and isinstance(error, UserSimulationRejected)
        ) or isinstance(error, (SlotRecoveryExhausted, QueueWaitError)) or (
            isinstance(error, RolloutInfrastructureError)
            and interaction_retryable(error.phase, error)
        )
        report = {
            "kind": "transient_exhausted" if transient else "fatal",
            "uid": uid,
            "partition_id": partition_id,
            "session_id": session_id,
            "phase": getattr(error, "phase", "agent_loop"),
            "error_type": type(error).__name__,
            "message": str(error),
        }
        self.failure_budget.call("record_rollout_failure", report)
        logger.error(
            "rollout failure: %s",
            report,
            exc_info=(type(error), error, error.__traceback__),
        )

    async def _run_agent_loop(self, sampling_params, **kwargs):
        partition = "val" if kwargs["trajectory"]["validate"] else "train"
        try:
            raise_if_fatal(self.failure_budget.call("fatal_rollout"))
            return await super()._run_agent_loop(sampling_params, **kwargs)
        except Exception as error:
            self._report_failure(
                error,
                uid=kwargs["uid"],
                partition_id=partition,
                session_id=kwargs["session_id"],
            )
            raise

    async def _run_prompt(self, prompt, sampling_params, trajectory, trace=False):
        uid = prompt["uid"]
        partition = "val" if trajectory["validate"] else "train"
        tasks = []
        try:
            raise_if_fatal(self.failure_budget.call("fatal_rollout"))
            await tq.async_kv_put(
                key=uid, partition_id=partition, tag={"status": "running"}
            )
            config = self.config.actor_rollout_ref.rollout
            n = prompt.pop(
                "__rollout_n__",
                config.val_kwargs.n if trajectory["validate"] else config.n,
            )
            if type(n) is not int or n < 1:
                raise ValueError("rollout group size must be a positive integer")
            params = dict(sampling_params)
            if not prompt.pop("__do_sample__", True) and not trajectory["validate"]:
                params.update(top_p=1.0, top_k=-1, temperature=0)
            tasks = [
                asyncio.create_task(
                    self._run_agent_loop(
                        params,
                        trajectory=trajectory,
                        trace=trace,
                        session_id=i,
                        **prompt,
                    )
                )
                for i in range(n)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            errors = [result for result in results if isinstance(result, BaseException)]
            failed_sessions = [i for i, result in enumerate(results) if isinstance(result, BaseException)]
            # A self-cancelled session is unexpected unless the prompt itself is
            # cancelled. Keep it visible instead of silently dropping a group.
            for error in errors:
                if isinstance(error, asyncio.CancelledError):
                    self._report_failure(error, uid=uid, partition_id=partition)
            await tq.async_kv_put(
                key=uid,
                partition_id=partition,
                tag={
                    "status": "failure" if errors else "finished",
                    "rollout_n": n,
                    "failed_session_ids": failed_sessions,
                    "failure_kind": (
                        "fatal" if self.failure_budget.call("fatal_rollout") else "transient_exhausted"
                    ) if errors else None,
                },
            )
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        except Exception as error:
            self._report_failure(error, uid=uid, partition_id=partition)
            await asyncio.gather(*tasks, return_exceptions=True)
            await tq.async_kv_put(
                key=uid, partition_id=partition, tag={"status": "failure"}
            )


class FailureAwareAgentLoopManagerTQ(AgentLoopManagerTQ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # create() initializes workers after construction, so replace the pinned
        # worker chosen by the parent's constructor before that initialization.
        self.agent_loop_workers_class = FailureAwareAgentLoopWorkerTQ

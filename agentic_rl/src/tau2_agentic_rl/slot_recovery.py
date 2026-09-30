"""Bounded recovery inside one original rollout slot and policy lease."""

import asyncio
import errno
import socket
from copy import deepcopy
from uuid import uuid4

import anyio
import httpcore
import httpx

from tau2_agentic_rl.user_simulation import (
    UserSimulationRejected,
    filter_enabled,
    replacement_seed,
)

RECOVERABLE_INTERACTION_PHASES = frozenset({
    "environment_reset", "model_generation", "tau2_tool_step", "tau2_text_step",
})
NONRECOVERABLE_INTERACTION_PHASES = frozenset({
    "policy_version_alignment", "rollout_log_probs", "token_alignment",
    "prompt_initialization", "tool_result_alignment", "tool_parser",
    "observation_tokenization", "user_sim_context", "official_reward",
})


RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


def _transient_service_error(error: BaseException) -> bool:
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    if isinstance(error, socket.gaierror):
        return error.errno == socket.EAI_AGAIN
    if isinstance(error, OSError):
        return error.errno in {
            errno.EAGAIN, errno.ETIMEDOUT, errno.ENETUNREACH, errno.EHOSTUNREACH,
            errno.ECONNREFUSED, errno.ECONNRESET, errno.ECONNABORTED, errno.EPIPE,
        }
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in RETRYABLE_HTTP_STATUSES
    if isinstance(error, (httpx.TransportError, httpcore.NetworkError,
                          httpcore.TimeoutException, httpcore.RemoteProtocolError)):
        return not isinstance(error, (httpx.UnsupportedProtocol, httpx.LocalProtocolError))
    # Tau2's optional OpenAI/LiteLLM clients use the SDK's exception types.
    try:
        from openai import APIConnectionError, APIStatusError
    except ImportError:
        return False
    return isinstance(error, APIConnectionError) or (
        isinstance(error, APIStatusError) and error.status_code in RETRYABLE_HTTP_STATUSES
    )


def interaction_retryable(phase: str, error: BaseException | None = None) -> bool:
    """Require explicit transient evidence; unknown errors never reroll a slot."""
    if phase not in RECOVERABLE_INTERACTION_PHASES or error is None:
        return False

    def check(current, ancestors, transport_context=False):
        if id(current) in ancestors:
            return False
        children = [cause for cause in (current.__cause__, current.__context__) if cause is not None]
        if isinstance(current, (TimeoutError, httpx.TimeoutException, httpcore.TimeoutException)):
            # wait_for/AnyIO implement deadlines by cancelling their own scope.
            # Inspect any deeper error, but do not mistake that internal cancel
            # for an outer task cancellation (which remains non-retryable).
            children = [nested for child in children for nested in (
                [cause for cause in (child.__cause__, child.__context__) if cause is not None]
                if isinstance(child, asyncio.CancelledError) else [child]
            )]
        if isinstance(current, BaseExceptionGroup):
            children.extend(current.exceptions)
            wrapper = True
        elif isinstance(current, RolloutInfrastructureError):
            wrapper = current.phase in RECOVERABLE_INTERACTION_PHASES
        elif type(current) is RuntimeError:
            wrapper = True
        elif type(current).__module__ == "ray.exceptions" and hasattr(current, "cause"):
            # RayTaskError carries its actual service exception separately.
            children.append(current.cause)
            wrapper = True
        else:
            wrapper = False
        transient = _transient_service_error(current) or (
            transport_context and isinstance(current, (anyio.BrokenResourceError, anyio.EndOfStream))
        )
        if not transient and not (wrapper and children):
            return False
        transport_context = transport_context or isinstance(current, (
            httpx.TransportError, httpcore.NetworkError, httpcore.TimeoutException,
            httpcore.RemoteProtocolError,
        ))
        return all(check(child, ancestors | {id(current)}, transport_context) for child in children)

    return check(error, set())


class RolloutInfrastructureError(RuntimeError):
    """A failed interaction whose audit was saved before requesting replacement."""

    def __init__(self, phase: str, trajectory_id: str):
        self.phase, self.trajectory_id = phase, trajectory_id
        label = "prompt initialization" if phase == "prompt_initialization" else phase
        super().__init__(f"rollout infrastructure failure during {label}; audit record saved")


class SlotRecoveryExhausted(RuntimeError):
    """Stop this slot; its incomplete group must be isolated from good groups."""


def slot_metadata(kwargs: dict) -> dict:
    context = kwargs.get("extra_info", {}).get("slot_recovery")
    return {"slot_recovery": deepcopy(context)} if context is not None else {}


async def run_training_slot(generate, sampling_params, *, project: dict, kwargs: dict, stop_requested=None, on_attempt=None):
    """Keep uid/session, task, sampling parameters and policy version unchanged."""
    max_infra = project["slot_recovery"]["max_resamples"]
    max_user = project.get("user_sim_filter", {}).get("max_resamples", 2) if filter_enabled(project) else 0
    if any(type(n) is not int or n < 0 for n in (max_infra, max_user)):
        raise ValueError("slot replacement budgets must be nonnegative integers")
    first_id = kwargs.get("trajectory_id") or uuid4().hex
    extra = dict(kwargs.get("extra_info", {}) or {})
    base_seed = extra.get("environment_seed")
    if base_seed is None:
        base_seed = int(first_id[:8], 16) % (2**31 - 1)
    failures = []
    infra_count = user_count = 0
    attempt_kind = "initial"
    while True:
        if stop_requested is not None and stop_requested():
            raise asyncio.CancelledError
        current_id = first_id if not failures else uuid4().hex
        context = {
            "parent_trajectory_id": first_id, "attempt": len(failures),
            "infrastructure_replacements": infra_count, "user_replacements": user_count,
            "failed_attempts": deepcopy(failures),
        }
        current = {
            **kwargs, "trajectory_id": current_id,
            "extra_info": {
                **extra, "environment_seed": replacement_seed(base_seed, user_count),
                "user_sim_attempt": extra.get("user_sim_attempt", 0) + user_count,
                "user_sim_parent_id": first_id, "slot_recovery": context,
            },
        }
        try:
            if on_attempt is not None:
                await on_attempt(attempt_kind)
            output = await generate(sampling_params, **current)
            output.extra_fields["slot_recovery"] = context
            return output
        except UserSimulationRejected as exc:
            failures.append({"trajectory_id": current_id, "phase": "invalid_user"})
            if user_count >= max_user:
                raise SlotRecoveryExhausted(f"user replacement limit in slot {first_id}") from exc
            user_count += 1
            attempt_kind = "user"
        except RolloutInfrastructureError as exc:
            # Unknown phases and broken training contracts must remain visible.
            if not interaction_retryable(exc.phase, exc):
                raise
            failures.append({"trajectory_id": exc.trajectory_id, "phase": exc.phase})
            if infra_count >= max_infra:
                raise SlotRecoveryExhausted(f"infrastructure replacement limit in slot {first_id}") from exc
            infra_count += 1
            attempt_kind = "infrastructure"

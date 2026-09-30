"""Bounded recovery inside one original rollout slot and policy lease."""

from copy import deepcopy
from uuid import uuid4

from tau2_agentic_rl.user_simulation import (
    UserSimulationRejected,
    filter_enabled,
    replacement_seed,
)

RECOVERABLE_INTERACTION_PHASES = frozenset({
    "environment_reset", "model_generation", "tau2_tool_step", "tau2_text_step",
})


class RolloutInfrastructureError(RuntimeError):
    """A failed interaction whose audit was saved before requesting replacement."""

    def __init__(self, phase: str, trajectory_id: str):
        self.phase, self.trajectory_id = phase, trajectory_id
        label = "prompt initialization" if phase == "prompt_initialization" else phase
        super().__init__(f"rollout infrastructure failure during {label}; audit record saved")


class SlotRecoveryExhausted(RuntimeError):
    """Stop instead of evicting good siblings or training an incomplete group."""


def slot_metadata(kwargs: dict) -> dict:
    context = kwargs.get("extra_info", {}).get("slot_recovery")
    return {"slot_recovery": deepcopy(context)} if context is not None else {}


async def run_training_slot(generate, sampling_params, *, project: dict, kwargs: dict):
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
    while True:
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
            output = await generate(sampling_params, **current)
            output.extra_fields["slot_recovery"] = context
            return output
        except UserSimulationRejected as exc:
            failures.append({"trajectory_id": current_id, "phase": "invalid_user"})
            if user_count >= max_user:
                raise SlotRecoveryExhausted(f"user replacement limit in slot {first_id}") from exc
            user_count += 1
        except RolloutInfrastructureError as exc:
            # Unknown phases and broken training contracts must remain visible.
            if exc.phase not in RECOVERABLE_INTERACTION_PHASES or isinstance(
                exc.__cause__, (ValueError, TypeError, KeyError, AssertionError),
            ):
                raise
            failures.append({"trajectory_id": exc.trajectory_id, "phase": exc.phase})
            if infra_count >= max_infra:
                raise SlotRecoveryExhausted(f"infrastructure replacement limit in slot {first_id}") from exc
            infra_count += 1

"""DAPO-style group filtering and GRPO advantage computation."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class ScoredRollout:
    """Minimal rollout information used during group selection."""

    task_id: str
    trajectory_id: str
    reward: float


@dataclass
class TrainingStepClock:
    """Keep candidate attempts separate from successful optimizer updates."""

    attempt_step: int = 0
    optimizer_step: int = 0
    consecutive_skips: int = 0

    def record_attempt(self, *, updated: bool) -> None:
        self.attempt_step += 1
        if updated:
            self.optimizer_step += 1
            self.consecutive_skips = 0
        else:
            self.consecutive_skips += 1


def is_mixed_reward_group(group: list[ScoredRollout], epsilon: float = 1e-12) -> bool:
    """Return true when a group contains usable reward variance."""
    return (
        bool(group)
        and max(item.reward for item in group) - min(item.reward for item in group)
        > epsilon
    )


def select_valid_groups(
    candidates: list[ScoredRollout],
    *,
    group_size: int = 8,
    target_groups: int = 4,
) -> list[list[ScoredRollout]]:
    """Select complete mixed groups without silently using partial groups."""
    by_task: dict[str, list[ScoredRollout]] = defaultdict(list)
    for rollout in candidates:
        by_task[rollout.task_id].append(rollout)
    valid = [
        group[:group_size]
        for group in by_task.values()
        if len(group) >= group_size and is_mixed_reward_group(group[:group_size])
    ]
    return valid[:target_groups] if len(valid) >= target_groups else []


def grpo_advantages(
    group: list[ScoredRollout], epsilon: float = 1e-6
) -> dict[str, float]:
    """Compute population-standardized trajectory-level GRPO advantages."""
    if not group:
        raise ValueError("cannot compute advantages for an empty group")
    rewards = [item.reward for item in group]
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    std = math.sqrt(variance)
    return {
        item.trajectory_id: (item.reward - mean) / (std + epsilon) for item in group
    }

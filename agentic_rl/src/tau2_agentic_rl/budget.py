"""Pre-generation hard context budget decisions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetDecision:
    """Result of a pre-generation context budget check."""

    can_generate: bool
    max_new_tokens: int
    remaining_tokens: int


@dataclass(frozen=True)
class ContextBudget:
    """Only the total context window limits a generation."""

    max_context_tokens: int = 16384

    @classmethod
    def from_rollout(cls, rollout):
        obsolete = {
            "reserved_observation_tokens", "reserved_template_tokens",
            "min_final_response_tokens", "per_turn_max_new_tokens",
        }.intersection(rollout)
        if obsolete:
            raise ValueError(f"remove obsolete per-turn limits/reserves: {sorted(obsolete)}")
        size = rollout["max_context_length"]
        if type(size) is not int or size < 1:
            raise ValueError("max_context_length must be a positive integer")
        if rollout.get("initial_prompt_max_tokens", size) != size:
            raise ValueError("initial prompt must use the same total context limit")
        return cls(max_context_tokens=size)

    def decide(self, current_context_tokens: int) -> BudgetDecision:
        """Choose the safe next-generation length before calling the model."""
        remaining = self.max_context_tokens - current_context_tokens
        if remaining <= 0:
            return BudgetDecision(False, 0, remaining)
        return BudgetDecision(
            True,
            remaining,
            remaining,
        )

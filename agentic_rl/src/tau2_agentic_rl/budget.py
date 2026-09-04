"""Pre-generation hard context budget decisions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetDecision:
    """Result of a pre-generation context budget check."""

    can_generate: bool
    max_new_tokens: int
    remaining_after_reserve: int


@dataclass(frozen=True)
class ContextBudget:
    """Token budget including reserves for environment and a final answer."""

    max_context_tokens: int = 16384
    reserved_observation_tokens: int = 1024
    reserved_template_tokens: int = 128
    min_final_response_tokens: int = 64
    per_turn_max_new_tokens: int = 1024

    def decide(self, current_context_tokens: int) -> BudgetDecision:
        """Choose the safe next-generation length before calling the model."""
        remaining = (
            self.max_context_tokens
            - current_context_tokens
            - self.reserved_observation_tokens
            - self.reserved_template_tokens
        )
        if remaining < self.min_final_response_tokens:
            return BudgetDecision(False, 0, remaining)
        return BudgetDecision(
            True,
            min(self.per_turn_max_new_tokens, remaining),
            remaining,
        )

"""Token/log-prob alignment checks for multi-turn actor rollouts."""

from __future__ import annotations

from collections.abc import Iterable

from tau2_agentic_rl.schemas import TokenTurn


def validate_turns(turns: Iterable[TokenTurn]) -> int:
    """Validate every raw generation and return total policy-token count."""
    turns = list(turns)
    indices = [turn.assistant_turn_index for turn in turns]
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError("assistant turn indices must be unique and ordered")
    return sum(len(turn.output_token_ids) for turn in turns)


def validate_aligned_response(
    response_ids: list[int],
    response_mask: list[int],
    aligned_old_log_probs: list[float],
    raw_turns: Iterable[TokenTurn],
) -> None:
    """Assert veRL response arrays and original generated fragments are aligned."""
    if not (len(response_ids) == len(response_mask) == len(aligned_old_log_probs)):
        raise ValueError(
            "response IDs, mask, and aligned log-probs must have equal length"
        )
    if any(value not in (0, 1) for value in response_mask):
        raise ValueError("response mask must be binary")
    policy_count = validate_turns(raw_turns)
    if sum(response_mask) != policy_count:
        raise ValueError("response mask does not select every original actor token")
    for mask, log_prob in zip(response_mask, aligned_old_log_probs, strict=True):
        if mask == 0 and log_prob != 0.0:
            raise ValueError("non-policy tokens must have zero aligned old log-prob")

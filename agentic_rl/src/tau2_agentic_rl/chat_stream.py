"""Budgeted environment turns shared by the live loop and tokenizer acceptance."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from tau2_agentic_rl.tooling import truncate_message_contents


async def render_environment_turn(
    messages: list[dict[str, Any]],
    *,
    tokenizer: Any,
    render: Callable[..., Awaitable[list[int]]],
    turn_separator: list[int],
    allowed_tokens: int,
    content_limit: int,
) -> tuple[list[dict[str, Any]], list[int], bool]:
    """Include separator tokens in every candidate's budget and zero-loss span."""
    low, high = 0, content_limit
    best = None
    while low <= high:
        middle = (low + high) // 2
        bounded, truncated = truncate_message_contents(messages, tokenizer, middle)
        rendered = await render(bounded, remove_system_prompt=True)
        ids = list(turn_separator) + rendered
        if len(ids) <= allowed_tokens:
            best = bounded, ids, truncated
            low = middle + 1
        else:
            high = middle - 1
    if best is None:
        raise RuntimeError("environment template and separator exceed token budget")
    return best

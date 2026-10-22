"""Budgeted environment turns shared by the live loop and tokenizer acceptance."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from tau2_agentic_rl.tool_result_limits import limit_tool_results
from tau2_agentic_rl.tooling import truncate_message_contents


async def render_environment_turn(
    messages: list[dict[str, Any]],
    *,
    tokenizer: Any,
    render: Callable[..., Awaitable[list[int]]],
    turn_separator: list[int],
    allowed_tokens: int,
    content_limit: int,
    tool_result_max_tokens: int | None = None,
) -> tuple[list[dict[str, Any]], list[int], bool]:
    """Include separator tokens in every candidate's budget and zero-loss span."""
    prepared, truncated = limit_tool_results(messages, tokenizer, tool_result_max_tokens)
    # Keep the existing user/synthetic observation cap separate from tool results.
    other_indices = [i for i, m in enumerate(prepared) if m.get("role") != "tool"]
    others, other_truncated = truncate_message_contents(
        [prepared[i] for i in other_indices], tokenizer, content_limit,
    )
    for index, message in zip(other_indices, others, strict=True):
        prepared[index] = message
    truncated |= other_truncated
    rendered = await render(prepared, remove_system_prompt=True)
    ids = list(turn_separator) + rendered
    if len(ids) <= allowed_tokens:
        return prepared, ids, truncated

    # Only the actual context/response capacity can force truncation by default.
    low, high = 0, sum(
        len(tokenizer.encode(str(m.get("content", "")), add_special_tokens=False))
        for m in prepared
    )
    best = None
    while low <= high:
        middle = (low + high) // 2
        bounded, hard_truncated = truncate_message_contents(prepared, tokenizer, middle)
        rendered = await render(bounded, remove_system_prompt=True)
        ids = list(turn_separator) + rendered
        if len(ids) <= allowed_tokens:
            best = bounded, ids, truncated or hard_truncated
            low = middle + 1
        else:
            high = middle - 1
    if best is None:
        raise RuntimeError("environment template and separator exceed token budget")
    return best

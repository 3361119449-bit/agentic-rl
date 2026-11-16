"""Complete environment turns; never clip user messages or tool receipts."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import Any


class ContextWindowExceeded(OverflowError):
    def __init__(self, required_tokens, available_tokens):
        self.required_tokens = required_tokens
        self.available_tokens = available_tokens
        super().__init__(
            f"complete environment turn and separator need {required_tokens} tokens; "
            f"only {available_tokens} remain in the context"
        )

    def __reduce__(self):
        return type(self), (self.required_tokens, self.available_tokens), self.__dict__


async def render_environment_turn(
    messages: list[dict[str, Any]],
    *,
    tokenizer: Any,
    render: Callable[..., Awaitable[list[int]]],
    turn_separator: list[int],
    allowed_tokens: int,
) -> tuple[list[dict[str, Any]], list[int], bool]:
    """Append all content, including separators, or report context exhaustion."""
    complete = deepcopy(messages)
    rendered = await render(complete, remove_system_prompt=True)
    ids = list(turn_separator) + rendered
    if len(ids) > allowed_tokens:
        raise ContextWindowExceeded(len(ids), allowed_tokens)
    return complete, ids, False

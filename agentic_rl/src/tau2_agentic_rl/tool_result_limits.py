"""Optional per-tool content limits shared by RL training and evaluation."""

from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any


def validate_tool_result_limit(value: Any) -> int | None:
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError("rollout.tool_result_max_tokens must be null or a positive integer")
    return value


def _parse_limit(value: str) -> int | None:
    try:
        return validate_tool_result_limit(None if value.lower() == "none" else int(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def add_tool_result_limit_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tool-result-max-tokens", type=_parse_limit, default=argparse.SUPPRESS,
        help="Per-tool result content cap; 'none' disables it (default in shipped configs). Model context limits still apply.",
    )


def configure_tool_result_limit(project: dict, args: argparse.Namespace) -> None:
    rollout = project.setdefault("rollout", {})
    if hasattr(args, "tool_result_max_tokens"):
        rollout["tool_result_max_tokens"] = args.tool_result_max_tokens
    validate_tool_result_limit(rollout.get("tool_result_max_tokens"))


def limit_tool_results(
    messages: list[dict[str, Any]], tokenizer: Any, max_tokens: int | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Cap each tool result only when requested; preserve the raw audit input."""
    # Config validation must not import tooling's reward/API dependency graph.
    from tau2_agentic_rl.tooling import truncate_message_contents

    validate_tool_result_limit(max_tokens)
    result = deepcopy(messages)
    truncated = False
    if max_tokens is not None:
        for index, message in enumerate(result):
            if message.get("role") == "tool":
                limited, changed = truncate_message_contents([message], tokenizer, max_tokens)
                result[index] = limited[0]
                truncated |= changed
    return result, truncated

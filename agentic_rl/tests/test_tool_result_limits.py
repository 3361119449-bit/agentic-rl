import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_rollout_integration import production_loop_class

from tau2_agentic_rl.budget import ContextBudget
from tau2_agentic_rl.chat_stream import render_environment_turn


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, tokens, **kwargs):
        return "".join(map(chr, tokens))


async def render(messages, **kwargs):
    return [1, 2] + [ord(c) for m in messages for c in m["content"]] + [3]


def test_default_preserves_large_tool_result_and_raw_audit():
    messages = [{"role": "tool", "tool_call_id": "a", "content": "x" * 3000}]
    original = deepcopy(messages)
    bounded, ids, truncated = asyncio.run(render_environment_turn(
        messages, tokenizer=Tokenizer(), render=render, turn_separator=[9],
        allowed_tokens=4000, content_limit=1024,
    ))
    assert bounded == messages == original
    assert len(ids) == 3004
    assert not truncated


@pytest.mark.parametrize("limit", [None, 40])
def test_limit_applies_per_tool_and_preserves_user_budget_and_ids(limit):
    messages = [
        {"role": "tool", "tool_call_id": "a", "content": "x" * 100},
        {"role": "tool", "tool_call_id": "b", "content": "y" * 100},
        {"role": "user", "content": "u" * 100},
    ]
    original = deepcopy(messages)
    bounded, ids, truncated = asyncio.run(render_environment_turn(
        messages, tokenizer=Tokenizer(), render=render, turn_separator=[9],
        allowed_tokens=1000, content_limit=20, tool_result_max_tokens=limit,
    ))
    assert [len(m["content"]) for m in bounded] == [limit or 100, limit or 100, 20]
    assert [m["tool_call_id"] for m in bounded[:2]] == ["a", "b"]
    assert truncated and ids[0] == 9
    assert messages == original


def test_unlimited_tools_still_respect_total_context_capacity():
    messages = [{"role": "tool", "content": "x" * 3000}]
    bounded, ids, truncated = asyncio.run(render_environment_turn(
        messages, tokenizer=Tokenizer(), render=render, turn_separator=[9],
        allowed_tokens=2000, content_limit=1024,
    ))
    assert truncated and len(ids) <= 2000
    assert len(bounded[0]["content"]) > 1024
    assert len(messages[0]["content"]) == 3000


@pytest.mark.parametrize("limit", [None, 50])
def test_production_loop_uses_configured_limit(limit):
    cls, _ = production_loop_class()
    loop = cls.__new__(cls)
    loop.project = {"rollout": {"tool_result_max_tokens": limit}}
    loop.budget = ContextBudget()
    loop.response_length = 16384
    loop.tokenizer, loop.turn_separator = Tokenizer(), [9]
    loop._render_full_chat = render
    bounded, ids, truncated = asyncio.run(loop._bounded_environment_messages(
        [{"role": "tool", "content": "x" * 3000}], [1] * 8000, [1] * 500,
    ))
    assert len(bounded[0]["content"]) == (limit or 3000)
    assert truncated is (limit is not None)
    assert len(ids) <= 8384


@pytest.mark.parametrize("module", ["train_airline_grpo", "evaluate_airline"])
@pytest.mark.parametrize("value, expected", [("none", None), ("2048", 2048)])
def test_both_launchers_offer_the_same_override(module, value, expected):
    from importlib import import_module

    from tau2_agentic_rl.tool_result_limits import configure_tool_result_limit

    args = import_module("scripts." + module).parse_args(["--tool-result-max-tokens", value])
    project = {"rollout": {"tool_result_max_tokens": 123}}
    configure_tool_result_limit(project, args)
    assert project["rollout"]["tool_result_max_tokens"] == expected


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "1024"])
def test_invalid_yaml_limit_is_rejected(limit):
    from tau2_agentic_rl.tool_result_limits import configure_tool_result_limit

    with pytest.raises(ValueError, match="tool_result_max_tokens"):
        configure_tool_result_limit({"rollout": {"tool_result_max_tokens": limit}}, SimpleNamespace())

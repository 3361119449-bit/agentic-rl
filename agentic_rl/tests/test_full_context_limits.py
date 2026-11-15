"""Exercise actual generation and observation budgets without a paid API."""

import asyncio
from copy import deepcopy

import pytest
from test_rollout_integration import minimal_loop

from tau2_agentic_rl.budget import ContextBudget


@pytest.mark.parametrize("used", [4000, 15197, 16383])
def test_generation_can_use_all_remaining_context(used):
    decision = ContextBudget().decide(used)
    assert decision.can_generate
    assert decision.max_new_tokens == 16384 - used


def test_only_a_full_context_blocks_generation():
    assert not ContextBudget().decide(16384).can_generate


@pytest.mark.parametrize("key", [
    "per_turn_max_new_tokens", "reserved_observation_tokens",
    "reserved_template_tokens", "min_final_response_tokens",
])
def test_old_config_limits_cannot_silently_return(key):
    with pytest.raises(ValueError, match="obsolete"):
        ContextBudget.from_rollout({"max_context_length": 16384, key: 1024})


class Tokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))


def observation_loop(scratch_dir):
    loop, _ = minimal_loop(scratch_dir)
    loop.tokenizer = Tokenizer()
    loop.response_length = 16384
    loop.turn_separator = [8]

    async def render(messages, **kwargs):
        return [7, *loop.tokenizer.encode(messages[0]["content"]), 9]

    loop._render_full_chat = render
    return loop


def test_live_environment_preserves_a_receipt_longer_than_1024(scratch_dir):
    loop = observation_loop(scratch_dir)
    messages = [{"role": "tool", "content": "abc" * 1000}]
    original = deepcopy(messages)
    result, ids, clipped = asyncio.run(
        loop._bounded_environment_messages(messages, [0] * 5000, [])
    )
    assert result == original == messages
    assert ids == [8, 7, *loop.tokenizer.encode(messages[0]["content"]), 9]
    assert not clipped


def test_receipt_overflow_stops_instead_of_clipping(scratch_dir):
    loop = observation_loop(scratch_dir)
    messages = [{"role": "tool", "content": "abc" * 1000}]
    original = deepcopy(messages)
    with pytest.raises(OverflowError):
        asyncio.run(loop._bounded_environment_messages(messages, [0] * 16360, []))
    assert messages == original

"""Real tokenizer IDs, not character-token stubs; optional CPU integration suite."""

import asyncio
import os
from pathlib import Path

import pytest

from tau2_agentic_rl.chat_stream import render_environment_turn


@pytest.fixture(scope="module")
def tokenizer():
    source = os.environ.get("QWEN_TOKENIZER_PATH")
    if not source:
        pytest.skip(
            "set QWEN_TOKENIZER_PATH to the local Qwen3-4B-Instruct-2507 tokenizer"
        )
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(source, local_files_only=True)


def encode(tokenizer, messages, generation=True):
    tools = (
        [
            {
                "type": "function",
                "function": {
                    "name": "get_reservation_details",
                    "description": "Get reservation",
                    "parameters": {
                        "type": "object",
                        "properties": {"reservation_id": {"type": "string"}},
                        "required": ["reservation_id"],
                    },
                },
            }
        ]
        if messages[0]["role"] == "system"
        else None
    )
    return tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=True, add_generation_prompt=generation
    )


def boundary(tokenizer):
    # Same EOS-suffix contract as pinned veRL initialize_turn_separator.
    probe = encode(tokenizer, [{"role": "user", "content": "x"}], False)
    eos = max(i for i, token in enumerate(probe) if token == tokenizer.eos_token_id)
    return probe[eos + 1 :]


async def append_observation(
    tokenizer, history, stream, assistant, observation, budget=4096, limit=4096
):
    separator = boundary(tokenizer)
    completed = encode(tokenizer, history + [assistant], False)
    assert separator
    assert completed[-len(separator) :] == separator
    # Generation stops on assistant EOS, omitting the template's newline.
    generated = completed[len(stream) : -len(separator)]
    assert generated[-1] == tokenizer.eos_token_id
    assert completed[: len(stream)] == stream

    async def render(messages, remove_system_prompt):
        assert remove_system_prompt
        return encode(tokenizer, messages)

    bounded, observation_ids, truncated = await render_environment_turn(
        [observation],
        tokenizer=tokenizer,
        render=render,
        turn_separator=separator,
        allowed_tokens=budget,
        content_limit=limit,
    )
    result = stream + generated + observation_ids
    history = history + [assistant] + bounded
    assert result == encode(tokenizer, history)
    assert len(observation_ids) <= budget
    return history, result, truncated


@pytest.mark.parametrize(
    "kind", ["tool", "user", "synthetic_error", "two_tools", "truncated"]
)
def test_incremental_tokens_equal_full_qwen_history(tokenizer, kind):
    history = [
        {"role": "system", "content": "You are an airline agent."},
        {"role": "user", "content": "Look up my booking."},
    ]
    stream = encode(tokenizer, history)
    assistant = {
        "role": "assistant",
        "content": '<tool_call>\n{"name":"get_reservation_details","arguments":{"reservation_id":"A"}}\n</tool_call>',
    }
    observation = {"role": "tool", "content": '{"reservation_id":"A"}'}
    if kind == "user":
        assistant = {"role": "assistant", "content": "Please confirm the cancellation."}
        observation = {"role": "user", "content": "Yes, cancel it."}
    if kind == "synthetic_error":
        assistant["content"] = "<tool_call>{bad JSON}</tool_call>"
        observation["content"] = '{"error":true,"type":"parse_error"}'
    if kind == "truncated":
        observation["content"] = "large observation " * 4000
    history, stream, truncated = asyncio.run(
        append_observation(
            tokenizer,
            history,
            stream,
            assistant,
            observation,
            budget=40 if kind == "truncated" else 4096,
        )
    )
    assert truncated == (kind == "truncated")
    if kind == "two_tools":
        asyncio.run(
            append_observation(tokenizer, history, stream, assistant, observation)
        )


def test_separator_cannot_overflow_the_budget(tokenizer):
    async def render(messages, remove_system_prompt):
        return encode(tokenizer, messages)

    empty = [{"role": "tool", "content": ""}]
    with pytest.raises(RuntimeError, match="separator"):
        asyncio.run(
            render_environment_turn(
                empty,
                tokenizer=tokenizer,
                render=render,
                turn_separator=boundary(tokenizer),
                allowed_tokens=len(encode(tokenizer, empty)),
                content_limit=0,
            )
        )


def test_live_loop_uses_the_budgeted_renderer_after_successful_text_delivery():
    source = (
        Path(__file__).parents[1] / "src/tau2_agentic_rl/agent_loop/airline.py"
    ).read_text(encoding="utf-8")
    assert "turn_separator=self.turn_separator" in source
    assert source.index("await environment.step_text(decoded)") < source.index(
        "confirmation.observe_visible_assistant_text"
    )
    assert "validate_tool_turn(decoded, len(calls))" in source

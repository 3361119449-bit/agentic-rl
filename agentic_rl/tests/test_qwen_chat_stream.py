"""Real tokenizer IDs, not character-token stubs; optional CPU integration suite."""

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_rollout_integration import minimal_loop

from tau2_agentic_rl.chat_stream import render_environment_turn
from tau2_agentic_rl.environment.tau2_gym import Tau2GymAdapter
from tau2_agentic_rl.initial_prompt import encode_full_chat
from tau2_agentic_rl.judge.prompts import build_judge_messages
from tau2_agentic_rl.reward.score import RewardConfig
from tau2_agentic_rl.schemas import JudgeResult


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
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=generation,
        return_dict=False,  # Match the pinned veRL wrapper on Transformers 4/5.
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
    assert source.index("await environment.step_text(content)") < source.index(
        "messages.extend(bounded_messages)"
    )
    assert "confirmation.authorize" not in source
    assert "validate_tool_turn(decoded, len(calls))" in source


def test_real_qwen_initial_prompt_is_encoded_without_left_truncation(tokenizer):
    messages = [
        {
            "role": "system",
            "content": "POLICY MUST SURVIVE " + "airline policy " * 5000,
        },
        {"role": "user", "content": "Please check the policy."},
    ]
    full = encode_full_chat(tokenizer, messages)
    expected = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    assert len(full) > 8192 and full == expected
    assert "POLICY MUST SURVIVE" in tokenizer.decode(full[:40])


def test_untruncated_encoder_rejects_template_truncation_override(tokenizer):
    with pytest.raises(ValueError, match="cannot override"):
        encode_full_chat(
            tokenizer,
            [{"role": "user", "content": "hi"}],
            template_kwargs={"truncation": True},
        )


@pytest.fixture
def text_loop(tokenizer, scratch_dir):
    """Real actor/adapter/renderers; only inference and Tau2 backend are fake."""
    loop, scope = minimal_loop(scratch_dir)
    captured = {"actions": [], "prompts": [], "judge": []}

    class Message(SimpleNamespace):
        def model_dump(self, **kwargs):
            return vars(self).copy()

    class Backend:
        def __init__(self):
            self.messages = [Message(role="user", content="Explain refunds.")]
            self.info = {"task": {}, "tools": []}
            self.done = False
            self._agent = SimpleNamespace(observation=self.messages)
            self._orchestrator = SimpleNamespace(
                trajectory=self.messages,
                environment=SimpleNamespace(get_db_hash=lambda: "unchanged"),
            )
            self._simulation_done = SimpleNamespace(is_set=lambda: self.done)

        def step(self, action):
            if action == "###STOP###":
                self.done = True
            else:
                captured["actions"].append(action)
                self.messages.append(Message(role="assistant", content=action))
                self.done = len(captured["actions"]) == 2
                if not self.done:
                    self.messages.append(
                        Message(role="user", content="Please continue.")
                    )
            if self.done:
                self.info["simulation_run"] = {"termination_reason": "user_stop"}
            return None, 0.0, self.done, False, self.info

    backend = Backend()

    class Environment(Tau2GymAdapter):
        async def reset(self, seed=None):
            self.env, self.info = backend, backend.info
            self._initial_db_hash = self.db_hash()
            return self._take_new_observations()

    async def parse(ids, schemas):
        # Parser text is not assumed to be sanitized. Actual EOS ID defines the
        # transport boundary, independently of the parser's text return value.
        return tokenizer.decode(ids), []

    async def snapshot(*args):
        return {}

    class Judge:
        async def evaluate(self, **inputs):
            captured["judge"] = build_judge_messages(**inputs)
            return JudgeResult(), "fixture", "fixture", "fixture"

    scope["Tau2GymAdapter"] = Environment
    loop.project["project"].update(tau2_commit="fixture", verl_commit="fixture")
    loop.tokenizer = tokenizer
    loop.apply_chat_template_kwargs = {}
    loop.system_prompt = []
    loop.turn_separator = boundary(tokenizer)
    loop.response_length = 16384
    loop.tool_parser = SimpleNamespace(extract_tool_calls=parse)
    loop.shared_budget = SimpleNamespace(acall=snapshot)
    loop.judge, loop.reward_config = Judge(), RewardConfig()
    loop.semantic, loop.transfer, loop.policy_rules = {"0": {}}, {"0": {}}, {"0": {}}
    loop.required_actions, loop.action_dependencies = {"0": []}, {}
    return loop, backend, captured


@pytest.mark.parametrize("evaluation", [False, True], ids=["rl", "evaluation"])
def test_text_eos_never_reaches_tau2_or_judge_but_training_stream_is_intact(
    tokenizer, text_loop, monkeypatch, evaluation
):
    loop, backend, captured = text_loop
    if evaluation:
        monkeypatch.setenv("EVALUATION_MANIFEST_ID", "offline-fixture")
    else:
        monkeypatch.delenv("EVALUATION_MANIFEST_ID", raising=False)
    replies = ["Let me check.", "Your refund is being processed. 退款正在处理中。"]
    generated = [
        tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
        for text in replies
    ]
    old_logprobs = [[-0.01 * (i + 1) for i in range(len(ids))] for ids in generated]

    async def generate(**kwargs):
        index = len(captured["prompts"])
        expected_history = [
            {"role": "system", "content": loop.agent_system_prompt},
            *[message.model_dump() for message in backend.messages],
        ]
        # Check the actual NEXT generation input, not just saved display text.
        assert kwargs["prompt_ids"] == encode_full_chat(
            tokenizer, expected_history, tools=[]
        )
        if evaluation:
            assert kwargs["sampling_params"]["seed"] == 42 + index * 100003
        else:
            assert "seed" not in kwargs["sampling_params"]
        captured["prompts"].append(list(kwargs["prompt_ids"]))
        return SimpleNamespace(
            token_ids=generated[index], log_probs=old_logprobs[index], extra_fields={}
        )

    loop.server_manager = SimpleNamespace(generate=generate)
    output = asyncio.run(
        loop._run_trajectory({}, extra_info={"task_id": "0", "environment_seed": 42})
    )
    record = next(loop.store.records())
    assert record.termination_reason == "user_stop"
    assert captured["actions"] == replies
    assert "<|im_end|>" not in str(record.environment_transcript)
    assert "<|im_end|>" not in str(captured["judge"])
    assert record.scoring_inputs["judge"]["trajectory"]["messages"] == (
        record.environment_transcript
    )
    assistant_messages = [m for m in record.messages if m["role"] == "assistant"]
    for index, (message, turn) in enumerate(
        zip(assistant_messages, record.token_turns, strict=True)
    ):
        assert message["content"] == replies[index]
        assert message["raw_generated_text"] == replies[index] + "<|im_end|>"
        assert turn.prompt_token_ids == captured["prompts"][index]
        assert turn.output_token_ids == generated[index]
        assert turn.output_old_log_probs == old_logprobs[index]
    assert output.prompt_ids + output.response_ids == (
        captured["prompts"][-1] + generated[-1]
    )
    observation_length = (
        len(captured["prompts"][1]) - len(output.prompt_ids) - len(generated[0])
    )
    assert observation_length > 0
    assert output.response_mask == (
        [1] * len(generated[0]) + [0] * observation_length + [1] * len(generated[1])
    )
    assert output.response_logprobs == (
        old_logprobs[0] + [0.0] * observation_length + old_logprobs[1]
    )


@pytest.mark.parametrize("text", ["", " \n\t"], ids=["eos_only", "whitespace_only"])
def test_blank_completed_reply_is_local_model_error_not_user_api_failure(
    tokenizer, text_loop, text
):
    loop, _, captured = text_loop
    loop.hard_turn_limit = 1
    tokens = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]

    async def generate(**kwargs):
        return SimpleNamespace(
            token_ids=tokens, log_probs=[-1.0] * len(tokens), extra_fields={}
        )

    loop.server_manager = SimpleNamespace(generate=generate)
    output = asyncio.run(loop._run_trajectory({}, extra_info={"task_id": "0"}))
    record = next(loop.store.records())
    assert not captured["actions"]  # Only cleanup, never an empty Tau2 action.
    assert record.metadata["failure_phase"] is None
    assert record.termination_reason == "hard_turn_limit"
    assert record.tool_events[0].error_kind == "parse_error"
    assert record.environment_transcript == [
        {"role": "user", "content": "Explain refunds.", "turn_idx": 0}
    ]
    assert record.messages[2]["content"] == text
    assert record.messages[2]["raw_generated_text"] == text + "<|im_end|>"
    assert record.messages[3]["role"] == "tool"
    assert record.token_turns[0].output_token_ids == tokens
    assert output.response_ids[: len(tokens)] == tokens
    assert output.response_mask[: len(tokens)] == [1] * len(tokens)
    assert output.response_logprobs[: len(tokens)] == [-1.0] * len(tokens)


def test_only_actual_terminal_eos_is_removed_not_literal_marker_text(
    tokenizer, text_loop
):
    loop, _, captured = text_loop
    loop.hard_turn_limit = 1
    # Ordinary token pieces can spell a marker without emitting its special ID.
    # Global string replacement would silently alter this intentional content.
    pieces = ["The literal marker is ", "<", "|im_end|", ">"]
    tokens = [
        token
        for piece in pieces
        for token in tokenizer.encode(piece, add_special_tokens=False)
    ] + [tokenizer.eos_token_id]
    assert tokens.count(tokenizer.eos_token_id) == 1

    async def generate(**kwargs):
        return SimpleNamespace(
            token_ids=tokens, log_probs=[-1.0] * len(tokens), extra_fields={}
        )

    loop.server_manager = SimpleNamespace(generate=generate)
    asyncio.run(loop._run_trajectory({}, extra_info={"task_id": "0"}))
    record = next(loop.store.records())
    assert captured["actions"] == ["".join(pieces)]
    assert record.messages[2]["content"] == "".join(pieces)
    assert record.messages[2]["raw_generated_text"] == "".join(pieces) + "<|im_end|>"
    assert record.token_turns[0].output_token_ids == tokens


@pytest.mark.parametrize("valid", [False, True])
@pytest.mark.parametrize("reward_judge", [False, True])
@pytest.mark.parametrize("official_reward", [0, 1])
def test_complete_actor_screens_user_before_reward_and_never_returns_invalid_tokens(
    tokenizer, text_loop, valid, reward_judge, official_reward
):
    from test_user_simulation import SCENARIO, verdict

    from tau2_agentic_rl.user_simulation import (
        UserSimulationRejected,
        UserSimulationVerdict,
    )

    loop, backend, captured = text_loop
    original_step = backend.step

    def step(action):
        obs, _, done, truncated, info = original_step(action)
        return obs, float(official_reward), done, truncated, info

    backend.step = step
    loop.project["judge"] = {"enabled": reward_judge}
    loop.project["user_sim_filter"] = {"enabled": True, "max_resamples": 2}
    backend.info["task"] = {
        "user_scenario": SCENARIO,
        "evaluation_criteria": "HIDDEN_REFERENCE",
        "expected_db_state": "HIDDEN_REFERENCE",
    }
    backend._user = SimpleNamespace(
        global_simulation_guidelines="Actual guidelines",
        persona_config=SimpleNamespace(to_guidelines_text=lambda: ""),
    )
    captured["quality"] = []

    class QualityJudge:
        async def evaluate(self, **inputs):
            assert "HIDDEN_REFERENCE" not in str(inputs)
            assert not captured["judge"]  # Agent reward judging has not happened.
            captured["quality"].append(inputs)
            return (
                UserSimulationVerdict.model_validate(verdict(valid)),
                "raw",
                "hash",
                "cache",
            )

    loop.user_sim_judge = QualityJudge()
    if not reward_judge:
        loop.judge = None  # Any accidental evaluate() call now fails the test.
    tokens = tokenizer.encode("The request is complete.", add_special_tokens=False) + [
        tokenizer.eos_token_id
    ]

    async def generate(**kwargs):
        return SimpleNamespace(
            token_ids=tokens, log_probs=[-1.0] * len(tokens), extra_fields={}
        )

    loop.server_manager = SimpleNamespace(generate=generate)
    if not valid:
        with pytest.raises(UserSimulationRejected):
            asyncio.run(loop._run_trajectory({}, extra_info={"task_id": "0"}))
        assert not captured["judge"]
    else:
        output = asyncio.run(loop._run_trajectory({}, extra_info={"task_id": "0"}))
        assert output.response_ids
        assert bool(captured["judge"]) == reward_judge
        if not reward_judge:
            assert output.reward_score == official_reward
            assert output.extra_fields["reward_extra_info"] == {
                "train_reward": official_reward,
                "tau2_official_reward": official_reward,
            }
            assert "custom_strict_success" not in output.extra_fields
    records = list(loop.store.records())
    assert len(records) == 1
    record = records[0]
    assert record.user_sim_result["user_sim_valid"] == valid
    assert (
        record.official_scores.reward == official_reward
    )  # Outcome never selects users.
    assert (record.custom_reward is not None) == (valid and reward_judge)
    assert len(captured["quality"]) == 1
    assert len(captured["quality"][0]["inputs"]["user_replies"]) == 2
    assert record.token_turns[0].output_token_ids == tokens

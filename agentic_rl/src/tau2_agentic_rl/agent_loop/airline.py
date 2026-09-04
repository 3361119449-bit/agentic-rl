"""Token-preserving veRL AgentLoopBase integration with Tau2 Airline Gym."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from jsonschema import ValidationError, validate
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.base_tool import OpenAIFunctionToolSchema

from tau2_agentic_rl.annotations import load_task_mapping
from tau2_agentic_rl.budget import ContextBudget
from tau2_agentic_rl.config import load_runtime_config
from tau2_agentic_rl.environment.tau2_gym import Tau2GymAdapter
from tau2_agentic_rl.judge.client import DeepSeekJudge, JudgeConfig
from tau2_agentic_rl.reward.official_tau2 import parse_official_reward_info
from tau2_agentic_rl.reward.required_actions import (
    arguments_equal,
    load_required_actions,
)
from tau2_agentic_rl.reward.score import score_trajectory
from tau2_agentic_rl.schemas import TokenTurn, ToolEvent, TrajectoryRecord
from tau2_agentic_rl.storage import TrajectoryStore
from tau2_agentic_rl.token_alignment import validate_aligned_response


def _confirmed_by_latest_user(messages: list[dict[str, Any]]) -> bool:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = str(message.get("content", "")).strip().lower()
        return content == "yes" or content.startswith(("yes,", "yes "))
    return False


def _split(kwargs: dict[str, Any]) -> tuple[str, str, int]:
    extra = kwargs.get("extra_info", {}) or {}
    task_id = str(extra.get("task_id", kwargs.get("task_id", "")))
    split = str(extra.get("split", "train"))
    policy_version = int(extra.get("policy_version", 0))
    if not task_id:
        raise ValueError("Tau2 Airline rollout requires extra_info.task_id")
    return task_id, split, policy_version


def _safe_arguments(raw: str) -> tuple[dict[str, Any], str | None]:
    """Parse a tool argument object without allowing one bad call to crash logging."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, str(exc)
    if not isinstance(value, dict):
        return {}, "tool arguments are not a JSON object"
    return value, None


def _mark_repetition(event: ToolEvent, earlier: list[ToolEvent]) -> None:
    """Mark unchanged failed retries and successful no-progress duplicates."""
    for previous in reversed(earlier):
        if previous.name != event.name or not arguments_equal(
            previous.arguments,
            event.arguments,
        ):
            continue
        if not previous.success and not event.success:
            event.unchanged_retry = True
        if (
            previous.success
            and event.success
            and previous.db_effect is False
            and event.db_effect is False
            and previous.result == event.result
        ):
            event.no_progress = True
        return


@register("tau2_airline")
class Tau2AirlineAgentLoop(AgentLoopBase):
    """Run one isolated Tau2 trajectory and return aligned veRL tokens."""

    def __init__(self, *args: Any, project_config_path: str, **kwargs: Any):
        super().__init__(*args, **kwargs)
        config_path = Path(project_config_path).resolve()
        self.project = load_runtime_config(config_path)
        self.root = config_path.parent.parent.parent
        annotations = self.project["annotations"]
        self.required_actions = load_required_actions(
            self.root / annotations["required_actions"]
        )
        self.semantic = load_task_mapping(self.root / annotations["semantic_checks"])
        self.transfer = load_task_mapping(self.root / annotations["transfer_rules"])
        self.policy_rules = load_task_mapping(self.root / annotations["policy_rules"])
        self.store = TrajectoryStore(
            self.root / self.project["outputs"]["trajectories"]
        )
        self.tool_parser = ToolParser.get_tool_parser("qwen3_coder", self.tokenizer)
        rollout = self.project["rollout"]
        self.budget = ContextBudget(
            max_context_tokens=int(rollout["max_context_length"]),
            reserved_observation_tokens=int(rollout["reserved_observation_tokens"]),
            reserved_template_tokens=int(rollout["reserved_template_tokens"]),
            min_final_response_tokens=int(rollout["min_final_response_tokens"]),
            per_turn_max_new_tokens=int(rollout["per_turn_max_new_tokens"]),
        )
        judge_config = self.project["judge"]
        self.judge = DeepSeekJudge(
            JudgeConfig(
                model=judge_config["model"],
                base_url=judge_config["base_url"],
                cache_dir=str(self.root / self.project["outputs"]["judge_cache"]),
                max_retries=int(judge_config["max_retries"]),
            )
        )
        self.hard_turn_limit = int(rollout["max_hard_turns"])
        self.response_length = int(self.rollout_config.response_length)

    async def run(
        self, sampling_params: dict[str, Any], **kwargs: Any
    ) -> AgentLoopOutput:
        task_id, split, policy_version = _split(kwargs)
        trajectory_id = uuid4().hex
        seed = kwargs.get("extra_info", {}).get("environment_seed")
        user = self.project["user_simulator"]
        environment = Tau2GymAdapter(
            task_id=task_id,
            user_model=user["model"],
            user_temperature=float(user["temperature"]),
            user_llm_args=user.get("llm_args", {}),
            user_cache_dir=self.root / self.project["outputs"]["user_cache"],
            user_max_retries=int(user.get("max_retries", 2)),
            max_steps=self.hard_turn_limit * 3,
        )
        incoming = await environment.reset(seed=seed)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": environment.policy},
            *incoming,
        ]
        schemas = environment.tool_schemas
        schemas_by_name = {
            item["function"]["name"]: item["function"]["parameters"] for item in schemas
        }
        parser_schemas = [OpenAIFunctionToolSchema(**item) for item in schemas]
        prompt_ids = await self.apply_chat_template(messages, tools=schemas)
        initial_prompt_ids = list(prompt_ids)
        response_mask: list[int] = []
        aligned_log_probs: list[float] = []
        token_turns: list[TokenTurn] = []
        tool_events: list[ToolEvent] = []
        assistant_turns = 0
        termination_reason = "environment_terminated"
        metrics = AgentLoopMetrics()
        request_id = trajectory_id
        terminated = False
        observed_policy_versions: set[int] = {policy_version}

        while not terminated:
            decision = self.budget.decide(len(prompt_ids))
            if not decision.can_generate:
                termination_reason = "budget_exhausted"
                break
            if assistant_turns >= self.hard_turn_limit:
                termination_reason = "hard_turn_limit"
                break

            turn_sampling = dict(sampling_params)
            turn_sampling["max_tokens"] = decision.max_new_tokens
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=turn_sampling,
            )
            rollout_step = (output.extra_fields or {}).get("max_global_steps")
            if rollout_step is not None:
                observed_policy_versions.add(int(rollout_step))
                policy_version = int(rollout_step)
            if len(observed_policy_versions - {0}) > 1:
                raise RuntimeError(
                    "one trajectory was generated by multiple policy versions"
                )
            if output.log_probs is None or len(output.token_ids) != len(
                output.log_probs
            ):
                raise RuntimeError(
                    "vLLM did not return aligned token-level old log-probs"
                )
            token_turns.append(
                TokenTurn(
                    assistant_turn_index=assistant_turns,
                    prompt_token_ids=list(prompt_ids),
                    output_token_ids=list(output.token_ids),
                    output_old_log_probs=list(output.log_probs),
                )
            )
            prompt_ids.extend(output.token_ids)
            response_mask.extend([1] * len(output.token_ids))
            aligned_log_probs.extend(output.log_probs)
            assistant_turns += 1

            content, calls = await self.tool_parser.extract_tool_calls(
                output.token_ids,
                parser_schemas,
            )
            decoded = self.tokenizer.decode(output.token_ids)
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": content if calls else decoded,
                "turn_idx": assistant_turns,
            }
            if calls:
                assistant_message["tool_calls"] = [
                    {
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                    for call in calls
                ]
            messages.append(assistant_message)

            if "<tool_call>" in decoded and not calls:
                tool_events.append(
                    ToolEvent(
                        event_id=f"{trajectory_id}:{len(tool_events)}",
                        sequence=len(tool_events),
                        turn_id=assistant_turns,
                        error_kind="parse_error",
                        result=decoded,
                    )
                )

            if calls:
                for extra_call in calls[1:]:
                    extra_arguments, extra_error = _safe_arguments(extra_call.arguments)
                    tool_events.append(
                        ToolEvent(
                            event_id=f"{trajectory_id}:{len(tool_events)}",
                            sequence=len(tool_events),
                            turn_id=assistant_turns,
                            name=extra_call.name,
                            arguments=extra_arguments,
                            success=False,
                            error_kind=("schema_invalid" if extra_error else None),
                            result="not executed: Tau2 policy permits one call per turn",
                        )
                    )
                call = calls[0]
                try:
                    arguments = json.loads(call.arguments)
                    if not isinstance(arguments, dict):
                        raise TypeError("tool arguments are not an object")
                    schema_error = False
                    if call.name in schemas_by_name:
                        try:
                            validate(arguments, schemas_by_name[call.name])
                        except ValidationError:
                            schema_error = True
                    step = await environment.step_tool(call.name, arguments)
                    unknown_tool = call.name not in environment.tool_names
                    event = ToolEvent(
                        event_id=f"{trajectory_id}:{len(tool_events)}",
                        sequence=len(tool_events),
                        turn_id=assistant_turns,
                        name=call.name,
                        arguments=arguments,
                        success=step.tool_success is True,
                        db_effect=step.db_changed,
                        confirmed_before=_confirmed_by_latest_user(messages),
                        error_kind=(
                            "unknown_tool"
                            if unknown_tool
                            else (
                                "schema_invalid"
                                if schema_error
                                else (
                                    None
                                    if step.tool_success is not False
                                    else "model_caused_execution_error"
                                )
                            )
                        ),
                        result=step.tool_result,
                    )
                except (json.JSONDecodeError, TypeError) as exc:
                    step = await environment.step_text(decoded)
                    event = ToolEvent(
                        event_id=f"{trajectory_id}:{len(tool_events)}",
                        sequence=len(tool_events),
                        turn_id=assistant_turns,
                        name=call.name,
                        error_kind="schema_invalid",
                        result=str(exc),
                    )
                _mark_repetition(event, tool_events)
                tool_events.append(event)
            else:
                step = await environment.step_text(decoded)

            messages.extend(step.messages)
            if step.messages:
                environment_ids = await self.apply_chat_template(
                    step.messages,
                    remove_system_prompt=True,
                )
                if (
                    len(prompt_ids) + len(environment_ids)
                    >= self.budget.max_context_tokens
                    or len(response_mask) + len(environment_ids) >= self.response_length
                ):
                    termination_reason = "budget_exhausted"
                    break
                prompt_ids.extend(environment_ids)
                response_mask.extend([0] * len(environment_ids))
                aligned_log_probs.extend([0.0] * len(environment_ids))
            terminated = step.terminated
            if terminated:
                tau_reason = environment.tau2_termination_reason()
                termination_reason = (
                    "user_stop" if tau_reason == "user_stop" else "agent_stop"
                )
                if any(
                    event.name == "transfer_to_human_agents" and event.success
                    for event in tool_events
                ):
                    termination_reason = "human_transfer"

        trajectory_for_judge = environment.full_trajectory()
        if not terminated:
            await environment.force_cleanup_stop()
        official_reward, official_payload = environment.official_reward_payload()
        if termination_reason in {"budget_exhausted", "hard_turn_limit"}:
            official_reward = 0.0
        official = parse_official_reward_info(official_reward, official_payload)

        semantic_row = self.semantic[task_id]
        transfer_rule = self.transfer[task_id]
        policy_row = self.policy_rules[task_id]
        judge_result, judge_raw, judge_prompt_hash = await self.judge.evaluate(
            task=environment.task,
            policy=environment.policy,
            trajectory={
                "messages": trajectory_for_judge,
                "tool_events": [event.model_dump() for event in tool_events],
                "termination_reason": termination_reason,
            },
            semantic_checks=semantic_row.get("semantic_checks", []),
            mandatory_policy_checks=policy_row.get("judge_checks", []),
            transfer_rule=transfer_rule,
        )
        required = self.required_actions[task_id]
        custom_reward = score_trajectory(
            events=tool_events,
            messages=messages,
            assistant_turns=assistant_turns,
            required_actions=required,
            official=official,
            judge=judge_result,
            transfer_rule=transfer_rule,
        )

        response_ids = prompt_ids[len(initial_prompt_ids) :]
        validate_aligned_response(
            response_ids,
            response_mask,
            aligned_log_probs,
            token_turns,
        )
        record = TrajectoryRecord(
            trajectory_id=trajectory_id,
            task_id=task_id,
            split=split,
            policy_version=policy_version,
            annotation_version=self.project["project"]["annotation_version"],
            reward_version=self.project["project"]["reward_version"],
            environment_seed=seed,
            termination_reason=termination_reason,
            assistant_turns=assistant_turns,
            trajectory_tokens=len(prompt_ids),
            messages=messages,
            tool_events=tool_events,
            token_turns=token_turns,
            initial_db_hash=environment.initial_db_hash(),
            final_db_hash=environment.db_hash(),
            official_scores=official,
            judge_result=judge_result,
            custom_reward=custom_reward,
            metadata={
                "judge_raw": judge_raw,
                "judge_prompt_hash": judge_prompt_hash,
                "user_prompt_hashes": environment.user_prompt_hashes(),
                "tau2_commit": self.project["project"]["tau2_commit"],
                "verl_commit": self.project["project"]["verl_commit"],
                "rollout_engine_version": os.environ.get("VLLM_VERSION", "unknown"),
            },
        )
        self.store.save(record)
        return AgentLoopOutput(
            prompt_ids=initial_prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=aligned_log_probs,
            reward_score=custom_reward.train_reward,
            num_turns=assistant_turns,
            metrics=metrics,
            extra_fields={
                "reward_extra_info": {
                    "train_reward": custom_reward.train_reward,
                    "tau2_official_reward": official.reward,
                    "custom_strict_success": custom_reward.strict_success,
                },
                "trajectory_id": trajectory_id,
                "task_id": task_id,
                "policy_version": policy_version,
                "tau2_official_reward": official.reward,
                "custom_strict_success": custom_reward.strict_success,
                "termination_reason": termination_reason,
            },
        )

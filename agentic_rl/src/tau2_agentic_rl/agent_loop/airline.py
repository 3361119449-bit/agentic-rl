"""Token-preserving veRL AgentLoopBase integration with Tau2 Airline Gym."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

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
from tau2_agentic_rl.chat_stream import render_environment_turn
from tau2_agentic_rl.concurrency import (
    BudgetLease,
    QueueWaitError,
    SharedBudget,
    limits_from_project,
    queue_options_from_project,
)
from tau2_agentic_rl.config import load_runtime_config
from tau2_agentic_rl.environment.tau2_gym import GymStep, Tau2GymAdapter
from tau2_agentic_rl.initial_prompt import (
    encode_full_chat,
    initial_messages,
    inspect_initial_prompt,
    require_initial_prompt_fits,
)
from tau2_agentic_rl.judge.client import DeepSeekJudge, JudgeConfig
from tau2_agentic_rl.judge.prompts import rubric_fingerprint
from tau2_agentic_rl.reward.required_actions import (
    arguments_equal,
    load_action_dependencies,
    load_required_actions,
)
from tau2_agentic_rl.reward.score import build_reward_config, score_trajectory
from tau2_agentic_rl.rollout_audit import audited_official_scores, snapshot_transcript
from tau2_agentic_rl.schemas import TokenTurn, ToolEvent, TrajectoryRecord
from tau2_agentic_rl.storage import TrajectoryStore
from tau2_agentic_rl.token_alignment import validate_aligned_response
from tau2_agentic_rl.tooling import (
    ConfirmationTracker,
    execute_validated_tool_call,
    synthetic_tool_error,
    validate_tool_call,
    validate_tool_turn,
)
from tau2_agentic_rl.versions import sha256_json


def _split(kwargs: dict[str, Any]) -> tuple[str, str, int]:
    extra = kwargs.get("extra_info", {}) or {}
    task_id = str(extra.get("task_id", kwargs.get("task_id", "")))
    split = str(extra.get("split", "train"))
    policy_version = int(extra.get("policy_version", 0))
    if not task_id:
        raise ValueError("Tau2 Airline rollout requires extra_info.task_id")
    return task_id, split, policy_version


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


def _local_step(environment: Tau2GymAdapter, message: dict[str, Any]) -> GymStep:
    """Return a synthetic observation without advancing the Tau2 backend."""
    return GymStep(
        messages=[message],
        reward=environment.last_reward,
        terminated=False,
        info=environment.info,
        db_changed=False,
        tool_success=False,
        tool_result=str(message.get("content", "")),
    )


def _termination_reason(tau_reason: str | None) -> str:
    """Preserve Tau2's specific termination reason for auditability."""
    known = {
        "user_stop",
        "agent_stop",
        "max_steps",
        "timeout",
        "too_many_errors",
        "agent_error",
        "user_error",
        "infrastructure_error",
        "context_window_exceeded",
        "unexpected_error",
    }
    return tau_reason if tau_reason in known else "environment_terminated"


@register("tau2_airline")
class Tau2AirlineAgentLoop(AgentLoopBase):
    """Run one isolated Tau2 trajectory and return aligned veRL tokens."""

    def __init__(self, *args: Any, project_config_path: str, **kwargs: Any):
        super().__init__(*args, **kwargs)
        config_path = Path(project_config_path).resolve()
        self.project = load_runtime_config(config_path)
        self.root = Path(
            os.environ.get(
                "AGENTIC_RL_PROJECT_ROOT", str(config_path.parent.parent.parent)
            )
        )
        annotations = self.project["annotations"]
        self.required_actions = load_required_actions(
            self.root / annotations["required_actions"]
        )
        self.action_dependencies = load_action_dependencies(
            self.root / annotations["action_dependencies"]
        )
        self.semantic = load_task_mapping(self.root / annotations["semantic_checks"])
        self.transfer = load_task_mapping(self.root / annotations["transfer_rules"])
        self.policy_rules = load_task_mapping(self.root / annotations["policy_rules"])
        self.store = TrajectoryStore(
            self.root / self.project["outputs"]["trajectories"]
        )
        rollout = self.project["rollout"]
        parser_name = str(rollout.get("tool_parser", "hermes"))
        if parser_name != "hermes":
            raise ValueError("Qwen3-4B-Instruct-2507 requires the hermes JSON parser")
        self.tool_parser = ToolParser.get_tool_parser(parser_name, self.tokenizer)
        self.reward_config = build_reward_config(self.project)
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
                provider=judge_config.get("provider", "DeepSeek"),
                base_url=judge_config["base_url"],
                cache_dir=str(self.root / self.project["outputs"]["judge_cache"]),
                max_retries=int(judge_config["max_retries"]),
            )
        )
        self.hard_turn_limit = int(rollout["max_hard_turns"])
        self.response_length = int(self.rollout_config.response_length)
        if self.processor is not None:
            raise ValueError("Tau2 Airline requires the text-only Qwen tokenizer")

    async def _render_full_chat(self, messages, tools=None, remove_system_prompt=False):
        ids = await asyncio.to_thread(
            encode_full_chat,
            self.tokenizer,
            messages,
            tools=tools,
            template_kwargs=self.apply_chat_template_kwargs,
        )
        if remove_system_prompt:
            prefix = list(self.system_prompt)
            if ids[: len(prefix)] != prefix:
                raise ValueError(
                    "environment turn does not match the template's system prefix"
                )
            ids = ids[len(prefix) :]
        return ids

    async def _bounded_environment_messages(
        self,
        raw_messages: list[dict[str, Any]],
        prompt_ids: list[int],
        response_mask: list[int],
    ) -> tuple[list[dict[str, Any]], list[int], bool]:
        """Always append a bounded observation after a side effect has occurred."""
        allowed = min(
            self.budget.max_context_tokens - len(prompt_ids),
            self.response_length - len(response_mask),
        )
        if allowed <= 0:
            raise RuntimeError("no token budget remains for environment observation")
        return await render_environment_turn(
            raw_messages,
            tokenizer=self.tokenizer,
            render=self._render_full_chat,
            turn_separator=self.turn_separator,
            allowed_tokens=allowed,
            content_limit=self.budget.reserved_observation_tokens,
        )

    async def run(
        self, sampling_params: dict[str, Any], **kwargs: Any
    ) -> AgentLoopOutput:
        self.shared_budget = SharedBudget(
            limits_from_project(self.project),
            require_ray=True,
            **queue_options_from_project(self.project),
        )
        task_id, split, policy_version = _split(kwargs)
        trajectory_id = (
            uuid4().hex
        )  # Allocate before queueing, including queue failures.
        acquired = False
        try:
            async with self.shared_budget.aslot("trajectories") as lease:
                acquired = True
                task = asyncio.create_task(
                    self._run_trajectory(
                        sampling_params,
                        trajectory_id=trajectory_id,
                        lease=lease,
                        **kwargs,
                    )
                )
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError:
                    # Tau2's synchronous user call may still be running in a
                    # thread. Retain the lease until the interaction really ends,
                    # even if shutdown sends more than one cancellation.
                    while not task.done():
                        try:
                            await asyncio.shield(task)
                        except asyncio.CancelledError:
                            continue
                        except Exception:
                            break
                    if not task.cancelled():
                        task.exception()
                    raise
        except QueueWaitError as exc:
            if acquired:
                raise
            self.store.save(
                TrajectoryRecord(
                    trajectory_id=trajectory_id,
                    task_id=task_id,
                    split=split,
                    policy_version=policy_version,
                    annotation_version=self.project["project"]["annotation_version"],
                    reward_version=self.project["project"]["reward_version"],
                    environment_seed=kwargs.get("extra_info", {}).get(
                        "environment_seed"
                    ),
                    termination_reason="infrastructure_error",
                    assistant_turns=0,
                    trajectory_tokens=0,
                    metadata={
                        "evaluation_sample_index": kwargs.get("extra_info", {}).get(
                            "evaluation_sample_index"
                        ),
                        "failure_phase": "trajectory_queue",
                        "failure_type": type(exc).__name__,
                        "failure_message": str(exc),
                        "queue": exc.details,
                        "concurrency": self.shared_budget.call("snapshot"),
                    },
                )
            )
            raise

    async def _run_trajectory(
        self,
        sampling_params: dict[str, Any],
        *,
        trajectory_id: str | None = None,
        lease: BudgetLease | None = None,
        **kwargs: Any,
    ) -> AgentLoopOutput:
        task_id, split, policy_version = _split(kwargs)
        trajectory_id = trajectory_id or uuid4().hex
        queue_wait_seconds = lease.queue_wait_seconds if lease is not None else 0.0

        def progress():
            if lease is not None:
                lease.progress()

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
        try:
            incoming = await environment.reset(seed=seed)
            progress()
        except Exception as exc:
            await environment.force_cleanup_stop()
            self.store.save(
                TrajectoryRecord(
                    trajectory_id=trajectory_id,
                    task_id=task_id,
                    split=split,
                    policy_version=policy_version,
                    annotation_version=self.project["project"]["annotation_version"],
                    reward_version=self.project["project"]["reward_version"],
                    environment_seed=seed,
                    termination_reason="infrastructure_error",
                    assistant_turns=0,
                    trajectory_tokens=0,
                    initial_db_hash=environment.initial_db_hash(),
                    final_db_hash=environment.safe_db_hash(),
                    metadata={
                        "evaluation_sample_index": kwargs.get("extra_info", {}).get(
                            "evaluation_sample_index"
                        ),
                        "failure_phase": "environment_reset",
                        "queue_wait_seconds": queue_wait_seconds,
                        "failure_type": type(exc).__name__,
                        "failure_message": str(exc),
                    },
                )
            )
            raise RuntimeError(
                "Tau2 environment reset failed; audit record saved"
            ) from exc
        messages = initial_messages(environment.policy, incoming)
        confirmation = ConfirmationTracker()
        prompt_measurement = None
        try:
            schemas = environment.tool_schemas
            schemas_by_name = {
                item["function"]["name"]: item["function"]["parameters"]
                for item in schemas
            }
            parser_schemas = [OpenAIFunctionToolSchema(**item) for item in schemas]
            prompt_ids = await self._render_full_chat(messages, tools=schemas)
            prompt_measurement = inspect_initial_prompt(
                task_id,
                len(prompt_ids),
                int(self.rollout_config.prompt_length),
                self.budget,
            )
            require_initial_prompt_fits(prompt_measurement)
        except Exception as exc:
            await environment.force_cleanup_stop()
            self.store.save(
                TrajectoryRecord(
                    trajectory_id=trajectory_id,
                    task_id=task_id,
                    split=split,
                    policy_version=policy_version,
                    annotation_version=self.project["project"]["annotation_version"],
                    reward_version=self.project["project"]["reward_version"],
                    environment_seed=seed,
                    termination_reason="infrastructure_error",
                    assistant_turns=0,
                    trajectory_tokens=(prompt_measurement or {}).get(
                        "initial_prompt_tokens", 0
                    ),
                    messages=messages,
                    initial_db_hash=environment.initial_db_hash(),
                    final_db_hash=environment.safe_db_hash(),
                    metadata={
                        "evaluation_sample_index": kwargs.get("extra_info", {}).get(
                            "evaluation_sample_index"
                        ),
                        "failure_phase": "prompt_initialization",
                        "initial_prompt": prompt_measurement,
                        "queue_wait_seconds": queue_wait_seconds,
                        "failure_type": type(exc).__name__,
                        "failure_message": str(exc),
                    },
                )
            )
            raise RuntimeError(
                "rollout prompt initialization failed; audit saved"
            ) from exc
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
        infrastructure_error: tuple[str, Exception] | None = None
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
            if os.environ.get("EVALUATION_MANIFEST_ID") and seed is not None:
                turn_sampling["seed"] = int(seed) + assistant_turns * 100003
            try:
                output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=turn_sampling,
                )
                progress()
            except Exception as exc:
                infrastructure_error = ("model_generation", exc)
                termination_reason = "infrastructure_error"
                break
            rollout_step = (output.extra_fields or {}).get("max_global_steps")
            if rollout_step is not None:
                observed_policy_versions.add(int(rollout_step))
                policy_version = int(rollout_step)
            if len(observed_policy_versions - {0}) > 1:
                infrastructure_error = (
                    "policy_version_alignment",
                    RuntimeError(
                        "one trajectory was generated by multiple policy versions"
                    ),
                )
                termination_reason = "infrastructure_error"
                break
            if output.log_probs is None or len(output.token_ids) != len(
                output.log_probs
            ):
                infrastructure_error = (
                    "rollout_log_probs",
                    RuntimeError(
                        "vLLM did not return aligned token-level old log-probs"
                    ),
                )
                termination_reason = "infrastructure_error"
                break
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

            decoded = self.tokenizer.decode(output.token_ids)
            if (
                not output.token_ids
                or output.token_ids[-1] != self.tokenizer.eos_token_id
            ):
                # Do not execute a partial tool call or invent an EOS to continue.
                messages.append(
                    {
                        "role": "assistant",
                        "content": decoded,
                        "turn_idx": assistant_turns,
                    }
                )
                termination_reason = "generation_truncated"
                break
            try:
                content, calls = await self.tool_parser.extract_tool_calls(
                    output.token_ids,
                    parser_schemas,
                )
            except Exception as exc:
                infrastructure_error = ("tool_parser", exc)
                termination_reason = "infrastructure_error"
                break
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": content if calls else decoded,
                "turn_idx": assistant_turns,
                "raw_generated_text": decoded,
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

            turn_error = validate_tool_turn(decoded, len(calls))
            if turn_error:
                event = ToolEvent(
                    event_id=f"{trajectory_id}:{len(tool_events)}",
                    sequence=len(tool_events),
                    turn_id=assistant_turns,
                    name=calls[0].name if calls else "",
                    error_kind=turn_error,
                    result=decoded,
                )
                tool_events.append(event)
                step = _local_step(
                    environment,
                    synthetic_tool_error(
                        "invalid_tool_turn",
                        turn_error,
                        "No call was executed. Emit one complete JSON tool call, "
                        "with no accompanying text or additional tool blocks.",
                    ),
                )
            elif calls:
                call = calls[0]
                checked = validate_tool_call(
                    call.name,
                    call.arguments,
                    schemas_by_name,
                    environment.tool_names,
                )
                if not checked.valid:
                    step = _local_step(
                        environment,
                        synthetic_tool_error(
                            call.name,
                            str(checked.error_kind),
                            str(checked.detail),
                        ),
                    )
                    event = ToolEvent(
                        event_id=f"{trajectory_id}:{len(tool_events)}",
                        sequence=len(tool_events),
                        turn_id=assistant_turns,
                        name=call.name,
                        arguments=checked.arguments,
                        error_kind=checked.error_kind,
                        result=checked.detail,
                    )
                else:
                    proposal = None
                    authorized = True
                    if call.name in {
                        "book_reservation",
                        "cancel_reservation",
                        "send_certificate",
                        "update_reservation_baggages",
                        "update_reservation_flights",
                        "update_reservation_passengers",
                    }:
                        authorized, proposal = confirmation.authorize(
                            call.name,
                            checked.arguments,
                        )
                    if not authorized:
                        detail = (
                            "Database write blocked. Present this exact action to the "
                            "user, obtain a new explicit confirmation, then retry it "
                            "without changing any argument."
                        )
                        step = _local_step(
                            environment,
                            synthetic_tool_error(
                                call.name, "confirmation_required", detail
                            ),
                        )
                        event = ToolEvent(
                            event_id=f"{trajectory_id}:{len(tool_events)}",
                            sequence=len(tool_events),
                            turn_id=assistant_turns,
                            name=call.name,
                            arguments=checked.arguments,
                            confirmed_before=False,
                            error_kind="confirmation_required",
                            result=detail,
                        )
                    else:
                        before_tool_hash = environment.safe_db_hash()
                        try:
                            step = await execute_validated_tool_call(
                                checked, environment.step_tool
                            )
                        except Exception as exc:
                            after_tool_hash = environment.safe_db_hash()
                            changed = (
                                before_tool_hash != after_tool_hash
                                if before_tool_hash is not None
                                and after_tool_hash is not None
                                else None
                            )
                            if proposal is not None and changed is not False:
                                confirmation.consume(proposal.proposal_hash)
                            tool_events.append(
                                ToolEvent(
                                    event_id=f"{trajectory_id}:{len(tool_events)}",
                                    sequence=len(tool_events),
                                    turn_id=assistant_turns,
                                    name=call.name,
                                    arguments=checked.arguments,
                                    success=False,
                                    db_effect=changed,
                                    confirmed_before=proposal is not None,
                                    confirmation_consumed=proposal is not None
                                    and changed is not False,
                                    confirmation_proposal_hash=proposal.proposal_hash
                                    if proposal
                                    else None,
                                    confirmation_turn_id=proposal.confirmation_turn_id
                                    if proposal
                                    else None,
                                    result=f"environment exception: {type(exc).__name__}: {exc}",
                                )
                            )
                            infrastructure_error = ("tau2_tool_step", exc)
                            termination_reason = "infrastructure_error"
                            break
                        event = ToolEvent(
                            event_id=f"{trajectory_id}:{len(tool_events)}",
                            sequence=len(tool_events),
                            turn_id=assistant_turns,
                            name=call.name,
                            arguments=checked.arguments,
                            success=step.tool_success is True,
                            db_effect=step.db_changed,
                            confirmed_before=(True if proposal is not None else None),
                            confirmation_proposal_hash=(
                                proposal.proposal_hash if proposal is not None else None
                            ),
                            confirmation_turn_id=(
                                proposal.confirmation_turn_id
                                if proposal is not None
                                else None
                            ),
                            error_kind=(
                                None
                                if step.tool_success is not False
                                else "model_caused_execution_error"
                            ),
                            result=step.tool_result,
                        )
                        if proposal is not None and (
                            event.success or event.db_effect is True
                        ):
                            confirmation.consume(proposal.proposal_hash)
                            event.confirmation_consumed = True
                _mark_repetition(event, tool_events)
                tool_events.append(event)
            else:
                try:
                    step = await environment.step_text(decoded)
                except Exception as exc:
                    infrastructure_error = ("tau2_text_step", exc)
                    termination_reason = "infrastructure_error"
                    break
                confirmation.observe_visible_assistant_text(decoded, assistant_turns)
                for index, reply in enumerate(step.messages):
                    if reply.get("role") == "user":
                        confirmation.observe_user_reply(
                            str(reply.get("content", "")), len(messages) + index
                        )

            progress()
            if step.messages:
                try:
                    (
                        bounded_messages,
                        environment_ids,
                        observation_truncated,
                    ) = await self._bounded_environment_messages(
                        step.messages, prompt_ids, response_mask
                    )
                except Exception as exc:
                    infrastructure_error = ("observation_tokenization", exc)
                    termination_reason = "infrastructure_error"
                    break
                messages.extend(bounded_messages)
                if observation_truncated and calls and tool_events:
                    tool_events[-1].observation_truncated = True
                prompt_ids.extend(environment_ids)
                response_mask.extend([0] * len(environment_ids))
                aligned_log_probs.extend([0.0] * len(environment_ids))
            terminated = step.terminated
            if terminated:
                tau_reason = environment.tau2_termination_reason()
                termination_reason = _termination_reason(tau_reason)
                if any(
                    event.name == "transfer_to_human_agents" and event.success
                    for event in tool_events
                ):
                    termination_reason = "human_transfer"

        trajectory_for_judge = snapshot_transcript(environment)
        interaction_termination_reason = termination_reason
        final_db_hash = environment.safe_db_hash()
        await environment.force_cleanup_stop()
        official = None
        try:
            official_reward, official_payload = environment.official_reward_payload()
            official = audited_official_scores(
                official_reward, official_payload, termination_reason
            )
        except Exception as exc:
            if infrastructure_error is None:
                infrastructure_error = ("official_reward", exc)
                termination_reason = "infrastructure_error"

        semantic_row = self.semantic[task_id]
        transfer_rule = self.transfer[task_id]
        policy_row = self.policy_rules[task_id]
        judge_result = None
        judge_raw = None
        judge_prompt_hash = None
        judge_cache_key = None
        scoring_inputs = {
            "judge": {
                "task": environment.task,
                "policy": environment.policy,
                "trajectory": {
                    "messages": trajectory_for_judge,
                    "tool_events": [event.model_dump() for event in tool_events],
                    "termination_reason": interaction_termination_reason,
                },
                "semantic_checks": semantic_row.get("semantic_checks", []),
                "mandatory_policy_checks": policy_row.get("judge_checks", []),
                "transfer_rule": transfer_rule,
            },
            "required_actions": self.required_actions[task_id],
            "official_scores": official.model_dump() if official is not None else None,
            "action_dependencies": self.action_dependencies.get(task_id, []),
            "reward_project_config": {
                "reward": self.project.get("reward", {}),
                "rollout": self.project["rollout"],
            },
        }
        if infrastructure_error is None:
            try:
                (
                    judge_result,
                    judge_raw,
                    judge_prompt_hash,
                    judge_cache_key,
                ) = await self.judge.evaluate(
                    **scoring_inputs["judge"],
                )
                progress()
            except Exception as exc:
                infrastructure_error = ("judge", exc)

        required = self.required_actions[task_id]
        custom_reward = None
        if infrastructure_error is None:
            try:
                if official is None or judge_result is None:
                    raise RuntimeError("successful rollout lacks scorer inputs")
                custom_reward = score_trajectory(
                    events=tool_events,
                    messages=trajectory_for_judge,
                    assistant_turns=assistant_turns,
                    required_actions=required,
                    official=official,
                    judge=judge_result,
                    transfer_rule=transfer_rule,
                    action_dependencies=self.action_dependencies.get(task_id, []),
                    config=self.reward_config,
                )
            except Exception as exc:
                infrastructure_error = ("reward_scoring", exc)

        response_ids = prompt_ids[len(initial_prompt_ids) :]
        try:
            validate_aligned_response(
                response_ids,
                response_mask,
                aligned_log_probs,
                token_turns,
            )
        except Exception as exc:
            if infrastructure_error is None or infrastructure_error[0] in {
                "judge",
                "reward_scoring",
            }:
                infrastructure_error = ("token_alignment", exc)
                termination_reason = "infrastructure_error"
                custom_reward = None
                judge_result = None
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
            environment_transcript=trajectory_for_judge,
            scoring_inputs=scoring_inputs,
            tool_events=tool_events,
            token_turns=token_turns,
            initial_db_hash=environment.initial_db_hash(),
            final_db_hash=final_db_hash,
            official_scores=official,
            judge_result=judge_result,
            custom_reward=custom_reward,
            metadata={
                "initial_prompt": prompt_measurement,
                "queue_wait_seconds": queue_wait_seconds,
                "interaction_termination_reason": interaction_termination_reason,
                "concurrency": await self.shared_budget.acall("snapshot"),
                "scoring_inputs_sha256": sha256_json(scoring_inputs),
                "evaluation_sample_index": kwargs.get("extra_info", {}).get(
                    "evaluation_sample_index"
                ),
                "judge_raw": judge_raw,
                "judge_prompt_hash": judge_prompt_hash,
                "judge_cache_key": judge_cache_key,
                "judge_rubric_sha256": rubric_fingerprint(
                    semantic_row.get("semantic_checks", []),
                    policy_row.get("judge_checks", []),
                    transfer_rule,
                ),
                "user_prompt_hashes": environment.user_prompt_hashes(),
                "tau2_commit": self.project["project"]["tau2_commit"],
                "verl_commit": self.project["project"]["verl_commit"],
                "rollout_engine_version": os.environ.get("VLLM_VERSION", "unknown"),
                "failure_phase": (
                    infrastructure_error[0] if infrastructure_error else None
                ),
                "failure_type": (
                    type(infrastructure_error[1]).__name__
                    if infrastructure_error
                    else None
                ),
                "failure_message": (
                    str(infrastructure_error[1]) if infrastructure_error else None
                ),
            },
        )
        self.store.save(record)
        if infrastructure_error is not None:
            phase, error = infrastructure_error
            raise RuntimeError(
                f"rollout infrastructure failure during {phase}; audit record saved"
            ) from error
        assert custom_reward is not None
        assert official is not None
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

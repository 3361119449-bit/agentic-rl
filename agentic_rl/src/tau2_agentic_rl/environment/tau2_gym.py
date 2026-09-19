"""Async adapter around the fixed Tau2 AgentGymEnv."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tau2_agentic_rl.environment.cached_user import build_cached_agent_gym_env


@dataclass
class ToolStepResult:
    """Per-call result from one assistant tool turn."""

    call_id: str
    name: str
    arguments: dict[str, Any]
    success: bool
    db_changed: bool | None
    result: str | None


@dataclass
class GymStep:
    """One Tau2 environment step normalized for the actor loop."""

    messages: list[dict[str, Any]]
    reward: float
    terminated: bool
    info: dict[str, Any]
    db_changed: bool
    tool_success: bool | None
    tool_result: str | None
    tool_results: list[ToolStepResult] | None = None


class Tau2GymAdapter:
    """Create one fresh Tau2 environment and database per trajectory."""

    def __init__(
        self,
        *,
        task_id: str,
        user_model: str,
        user_temperature: float = 0.0,
        user_llm_args: dict[str, Any] | None = None,
        user_cache_dir: str | Path = "outputs/user_cache",
        user_max_retries: int = 2,
        max_steps: int = 100,
    ):
        self.task_id = str(task_id)
        self.user_model = user_model
        self.user_temperature = user_temperature
        self.user_llm_args = dict(user_llm_args or {})
        self.user_cache_dir = Path(user_cache_dir)
        self.user_max_retries = user_max_retries
        self.max_steps = max_steps
        self.env: Any = None
        self.info: dict[str, Any] = {}
        self.last_reward = 0.0
        self._observation_count = 0
        self._initial_db_hash: str | None = None

    async def reset(self, seed: int | None = None) -> list[dict[str, Any]]:
        """Start a new isolated AgentGymEnv and return its initial user messages."""
        llm_args = {"temperature": self.user_temperature, **self.user_llm_args}
        self.env = build_cached_agent_gym_env(
            domain="airline",
            task_id=self.task_id,
            max_steps=self.max_steps,
            solo_mode=False,
            user_llm=self.user_model,
            user_llm_args=llm_args,
            user_cache_dir=self.user_cache_dir,
            user_max_retries=self.user_max_retries,
            all_messages_as_observation=False,
        )
        _, self.info = await asyncio.to_thread(self.env.reset, seed=seed)
        if getattr(self.env, "_simulation_done", None).is_set():
            raise RuntimeError("Tau2 environment terminated during reset")
        self._initial_db_hash = self.db_hash()
        return self._take_new_observations()

    @property
    def policy(self) -> str:
        return str(self.info["policy"])

    @property
    def task(self) -> dict[str, Any]:
        task = self.info["task"]
        if hasattr(task, "model_dump"):
            return task.model_dump(mode="json")
        return dict(task)

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return Tau2 tools in OpenAI function-tool schema form."""
        return [dict(tool.openai_schema) for tool in self.info["tools"]]

    @property
    def tool_names(self) -> set[str]:
        """Return the exact set of tools exposed by this environment."""
        return {str(tool.name) for tool in self.info["tools"]}

    def db_hash(self) -> str | None:
        """Read the current environment DB hash through Tau2's public method."""
        orchestrator = getattr(self.env, "_orchestrator", None)
        environment = getattr(orchestrator, "environment", None)
        if environment is None:
            return None
        value = environment.get_db_hash()
        return str(value) if value is not None else None

    def safe_db_hash(self) -> str | None:
        """Best-effort hash for failure-path audit records."""
        try:
            return self.db_hash()
        except Exception:
            return None

    async def step_text(self, content: str) -> GymStep:
        """Send one assistant text message to the Tau2 user simulator."""
        return await self._step(content)

    async def step_tool(self, name: str, arguments: dict[str, Any]) -> GymStep:
        """Execute one tool call through the native multi-tool path."""
        return await self.step_tools(
            [{"id": "call_0", "name": name, "arguments": arguments}]
        )

    async def step_tools(self, calls: list[dict[str, Any]]) -> GymStep:
        """Execute one assistant turn containing one or more Tau2 tool calls."""
        if not calls:
            raise ValueError("step_tools requires at least one tool call")
        return await asyncio.to_thread(self._step_tools_sync, calls)

    def _step_tools_sync(self, calls: list[dict[str, Any]]) -> GymStep:
        """Feed a native AssistantMessage to pinned Tau2 and audit every result."""
        # Keep this module importable in CPU-only checks that intentionally do
        # not install Tau2. The live rollout environment always has pinned Tau2.
        from tau2.data_model.message import AssistantMessage, ToolCall

        gym_env = self.env
        orchestrator = getattr(gym_env, "_orchestrator", None)
        agent = getattr(gym_env, "_agent", None)
        simulation_done = getattr(gym_env, "_simulation_done", None)
        lock = getattr(gym_env, "_lock", None)
        backend = getattr(orchestrator, "environment", None)
        if any(item is None for item in (orchestrator, agent, simulation_done, lock, backend)):
            raise RuntimeError("Tau2 gym is not initialized for tool execution")

        normalized = [
            {
                "id": str(call.get("id") or f"call_{index}"),
                "name": str(call["name"]),
                "arguments": dict(call["arguments"]),
            }
            for index, call in enumerate(calls)
        ]
        action = AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id=call["id"],
                    name=call["name"],
                    arguments=call["arguments"],
                    requestor="assistant",
                )
                for call in normalized
            ],
            raw_data={"action": "multi_tool_batch"},
        )

        audits: list[ToolStepResult] = []
        original_get_response = backend.get_response

        def audited_get_response(tool_call: ToolCall):
            before = backend.get_db_hash()
            tool_message = original_get_response(tool_call)
            after = backend.get_db_hash()
            before_hash = str(before) if before is not None else None
            after_hash = str(after) if after is not None else None
            audits.append(
                ToolStepResult(
                    call_id=str(tool_call.id),
                    name=str(tool_call.name),
                    arguments=dict(tool_call.arguments),
                    success=not bool(tool_message.error),
                    db_changed=(
                        before_hash != after_hash
                        if before_hash is not None and after_hash is not None
                        else None
                    ),
                    result=(
                        str(tool_message.content)
                        if tool_message.content is not None
                        else None
                    ),
                )
            )
            return tool_message

        before_hash = self.db_hash()
        backend.get_response = audited_get_response
        try:
            with lock:
                if simulation_done.is_set():
                    raise RuntimeError("Tau2 simulation already terminated")
                agent.set_action(action)
                while not simulation_done.is_set() and not agent.is_agent_turn:
                    simulation_done.wait(timeout=0.01)
                terminated = simulation_done.is_set()
                reward, reward_info = gym_env._get_reward()
                info = gym_env._get_info()
                info["reward_info"] = reward_info
        finally:
            backend.get_response = original_get_response

        self.info = info
        if terminated and not self._simulation_payload(info):
            raise RuntimeError("Tau2 orchestrator terminated without a simulation run")
        if len(audits) != len(normalized):
            raise RuntimeError(
                f"Tau2 returned {len(audits)} results for {len(normalized)} tool calls"
            )
        self.last_reward = float(reward)
        messages = self._take_new_observations()
        after_hash = self.db_hash()
        return GymStep(
            messages=messages,
            reward=float(reward),
            terminated=bool(terminated),
            info=info,
            db_changed=(
                before_hash is not None
                and after_hash is not None
                and before_hash != after_hash
            ),
            tool_success=all(item.success for item in audits),
            tool_result=audits[-1].result if audits else None,
            tool_results=audits,
        )

    async def _step(self, action: str) -> GymStep:
        before_hash = self.db_hash()
        _, reward, terminated, _, info = await asyncio.to_thread(self.env.step, action)
        self.info = info
        if terminated and not self._simulation_payload(info):
            raise RuntimeError("Tau2 orchestrator terminated without a simulation run")
        self.last_reward = float(reward)
        messages = self._take_new_observations()
        after_hash = self.db_hash()
        tool_messages = [item for item in messages if item.get("role") == "tool"]
        success: bool | None = None
        result: str | None = None
        if tool_messages:
            latest = tool_messages[-1]
            success = not bool(latest.get("error", False))
            result = str(latest.get("content", ""))
        return GymStep(
            messages=messages,
            reward=float(reward),
            terminated=bool(terminated),
            info=info,
            db_changed=(
                before_hash is not None
                and after_hash is not None
                and before_hash != after_hash
            ),
            tool_success=success,
            tool_result=result,
        )

    def _take_new_observations(self) -> list[dict[str, Any]]:
        agent = getattr(self.env, "_agent", None)
        raw_messages = list(getattr(agent, "observation", []) or [])
        new = raw_messages[self._observation_count :]
        self._observation_count = len(raw_messages)
        # The full observation also contains the externally supplied assistant
        # action. The actor loop already stores that exact generated message, so
        # only append new environment/user messages here.
        return [
            self._message_to_chat(item)
            for item in new
            if getattr(item, "role", None) != "assistant"
        ]

    @staticmethod
    def _message_to_chat(message: Any) -> dict[str, Any]:
        raw = message.model_dump(mode="json", exclude_none=True)
        result: dict[str, Any] = {
            "role": raw["role"],
            "content": raw.get("content", ""),
        }
        if "id" in raw and raw["role"] == "tool":
            result["tool_call_id"] = raw["id"]
        if "error" in raw:
            result["error"] = raw["error"]
        return result

    def full_trajectory(self) -> list[dict[str, Any]]:
        """Snapshot Tau2's full live trajectory for judging and auditing."""
        orchestrator = getattr(self.env, "_orchestrator", None)
        trajectory = list(getattr(orchestrator, "trajectory", []) or [])
        return [item.model_dump(mode="json", exclude_none=True) for item in trajectory]

    def initial_db_hash(self) -> str | None:
        """Return the actual live database hash captured immediately after reset."""
        return self._initial_db_hash

    async def force_cleanup_stop(self) -> None:
        """Release Tau2's waiting daemon after an external budget/turn stop."""
        if self.env is None:
            return
        simulation_done = getattr(self.env, "_simulation_done", None)
        if simulation_done is not None and simulation_done.is_set():
            return
        try:
            await self._step("###STOP###")
        except Exception:
            return

    def official_reward_payload(self) -> tuple[float, str | dict[str, Any]]:
        """Return final official reward data emitted by Tau2 Gym."""
        return self.last_reward, self.info.get("reward_info", {})

    def tau2_termination_reason(self) -> str | None:
        parsed = self._simulation_payload(self.info)
        return parsed.get("termination_reason") if parsed else None

    @staticmethod
    def _simulation_payload(info: dict[str, Any]) -> dict[str, Any]:
        raw = info.get("simulation_run")
        if not raw:
            return {}
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        return parsed if isinstance(parsed, dict) else {}

    def user_prompt_hashes(self) -> list[str]:
        """Return hashes for every user-simulator request in this trajectory."""
        user = getattr(self.env, "_user", None)
        return list(getattr(user, "prompt_hashes", []))

    def user_sim_context(self):
        """Only the actual simulator guidelines and scenario, never task criteria."""
        user = self.env._user
        guidelines = user.global_simulation_guidelines
        persona = user.persona_config.to_guidelines_text() or ""
        if persona:
            persona = f"\n\n{persona}\n"
        guidelines = guidelines.replace("<PERSONA_GUIDELINES>", persona)
        return guidelines, self.task["user_scenario"]

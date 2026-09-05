"""Cached, retry-bounded Tau2 user simulator isolated from the judge."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from tau2_agentic_rl.concurrency import api_budget
from tau2_agentic_rl.versions import sha256_json


def _serialize(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    return value


class CachedUserSimulatorMixin:
    """Mixin overriding Tau2 generation with a private deterministic cache."""

    def __init__(
        self, *args: Any, cache_dir: str | Path, max_retries: int = 2, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.prompt_hashes: list[str] = []

    def _generate_next_message(self, message: Any, state: Any) -> Any:
        from tau2.data_model.message import (
            AssistantMessage,
            MultiToolMessage,
            ToolCall,
            ToolMessage,
            UserMessage,
        )
        from tau2.utils.llm_utils import generate

        if isinstance(message, AssistantMessage) and message.is_audio:
            raise ValueError("cached text user simulator does not accept audio")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        elif isinstance(message, ToolMessage):
            state.messages.append(message)
        elif message.has_content() or message.is_tool_call():
            state.messages.append(message)

        messages = state.system_messages + state.flip_roles()
        cache_input = {
            "model": self.llm,
            "messages": [_serialize(item) for item in messages],
            "tools": [tool.openai_schema for tool in (self.tools or [])],
            "llm_args": self.llm_args,
        }
        prompt_hash = sha256_json(cache_input)
        self.prompt_hashes.append(prompt_hash)
        cache_path = self.cache_dir / f"{prompt_hash}.json"

        if cache_path.exists():
            assistant_message = AssistantMessage.model_validate_json(
                cache_path.read_text(encoding="utf-8")
            )
        else:
            last_error: Exception | None = None
            for attempt in range(self.max_retries + 1):
                try:
                    with api_budget().slot("user_api"):
                        assistant_message = generate(
                            model=self.llm,
                            messages=messages,
                            tools=self.tools,
                            call_name="user_simulator_response",
                            **self.llm_args,
                        )
                    tmp = cache_path.with_name(f".{cache_path.name}.{uuid4().hex}.tmp")
                    tmp.write_text(
                        assistant_message.model_dump_json(indent=2),
                        encoding="utf-8",
                    )
                    os.replace(tmp, cache_path)
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < self.max_retries:
                        time.sleep(2**attempt)
            else:
                raise RuntimeError(
                    "user simulator failed after bounded retries"
                ) from last_error

        user_message = UserMessage(
            role="user",
            content=assistant_message.content,
            cost=assistant_message.cost,
            usage=assistant_message.usage,
            raw_data=assistant_message.raw_data,
        )
        if assistant_message.tool_calls is not None:
            user_message.tool_calls = [
                ToolCall(
                    id=call.id,
                    name=call.name,
                    arguments=call.arguments,
                    requestor="user",
                )
                for call in assistant_message.tool_calls
            ]
        return user_message


def build_cached_agent_gym_env(**kwargs: Any) -> Any:
    """Construct an AgentGymEnv whose user has an isolated response cache."""
    from tau2.gym.gym_agent import AgentGymEnv
    from tau2.user.user_simulator import UserSimulator

    cache_dir = kwargs.pop("user_cache_dir")
    max_retries = int(kwargs.pop("user_max_retries", 2))

    class CachedUserSimulator(CachedUserSimulatorMixin, UserSimulator):
        pass

    class CachedAgentGymEnv(AgentGymEnv):
        def _get_user(self) -> Any:
            environment = self._get_environment()
            task = self._get_task()
            try:
                tools = environment.get_user_tools(include=task.user_tools) or None
            except ValueError:
                tools = None
            return CachedUserSimulator(
                tools=tools,
                instructions=task.user_scenario,
                llm=self.user_llm,
                llm_args=self.user_llm_args,
                cache_dir=cache_dir,
                max_retries=max_retries,
            )

    return CachedAgentGymEnv(**kwargs)

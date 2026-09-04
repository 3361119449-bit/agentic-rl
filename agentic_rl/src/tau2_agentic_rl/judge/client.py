"""Async OpenAI-compatible client used only for final trajectory judging."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from tau2_agentic_rl.judge.prompts import build_judge_messages
from tau2_agentic_rl.schemas import JudgeResult
from tau2_agentic_rl.versions import sha256_json


@dataclass(frozen=True)
class JudgeConfig:
    """External judge endpoint configuration."""

    model: str
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    timeout_seconds: float = 120.0
    max_retries: int = 2
    cache_dir: str = "outputs/judge_cache"


class DeepSeekJudge:
    """One isolated, cached judge call per complete trajectory."""

    def __init__(self, config: JudgeConfig):
        if config.model.startswith("FIX_EXACT_"):
            raise ValueError("set an exact judge model ID before running")
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_requested_criteria(
        result: JudgeResult, inputs: dict[str, Any]
    ) -> None:
        """Reject missing, duplicated, invented, or reordered rubric decisions."""

        expected_semantic = [
            str(item["criterion_id"]) for item in inputs.get("semantic_checks", [])
        ]
        expected_policy = [
            str(item.get("criterion_id", item.get("rule_id")))
            for item in inputs.get("mandatory_policy_checks", [])
        ]
        if "None" in expected_policy:
            raise ValueError("each mandatory judge check needs criterion_id or rule_id")
        actual_semantic = [item.criterion_id for item in result.semantic_checks]
        expected_transfer_semantic = [
            str(item["criterion_id"])
            for item in inputs.get("transfer_rule", {}).get("semantic_checks", [])
        ]
        actual_transfer_semantic = [
            item.criterion_id for item in result.transfer_semantic_checks
        ]
        actual_policy = [item.criterion_id for item in result.mandatory_policy_checks]
        if actual_semantic != expected_semantic:
            raise ValueError(
                "judge semantic criterion IDs do not exactly match the requested rubric: "
                f"expected={expected_semantic}, actual={actual_semantic}"
            )
        if actual_transfer_semantic != expected_transfer_semantic:
            raise ValueError(
                "judge transfer-semantic criterion IDs do not exactly match the "
                "requested rubric: "
                f"expected={expected_transfer_semantic}, "
                f"actual={actual_transfer_semantic}"
            )
        if actual_policy != expected_policy:
            raise ValueError(
                "judge policy criterion IDs do not exactly match the requested rubric: "
                f"expected={expected_policy}, actual={actual_policy}"
            )

        tool_events = inputs.get("trajectory", {}).get("tool_events", [])
        transferred = any(
            item.get("name") == "transfer_to_human_agents" and item.get("success")
            for item in tool_events
        )
        transfer_required = bool(inputs.get("transfer_rule", {}).get("required", False))
        expected_transfer_applicable = transferred or transfer_required
        if result.transfer_check.applicable != expected_transfer_applicable:
            raise ValueError(
                "judge transfer_check.applicable disagrees with the fixed trajectory/rubric"
            )
        if not transferred and result.transfer_check.valid:
            raise ValueError("judge cannot mark a non-executed transfer as valid")

    async def evaluate(self, **inputs: Any) -> tuple[JudgeResult, str, str]:
        """Return validated result, raw response, and prompt hash."""
        messages = build_judge_messages(**inputs)
        prompt_hash = sha256_json(messages)
        cache_path = self.cache_dir / f"{prompt_hash}.json"
        if cache_path.exists():
            raw = cache_path.read_text(encoding="utf-8")
            parsed = JudgeResult.model_validate_json(raw)
            self._validate_requested_criteria(parsed, inputs)
            return parsed, raw, prompt_hash

        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing {self.config.api_key_env}")
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        last_error: Exception | None = None
        async with httpx.AsyncClient(
            base_url=self.config.base_url.rstrip("/") + "/",
            timeout=self.config.timeout_seconds,
        ) as client:
            for attempt in range(self.config.max_retries + 1):
                try:
                    response = await client.post(
                        "chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json=payload,
                    )
                    response.raise_for_status()
                    raw = response.json()["choices"][0]["message"]["content"]
                    parsed = JudgeResult.model_validate(json.loads(raw))
                    self._validate_requested_criteria(parsed, inputs)
                    cache_path.write_text(
                        parsed.model_dump_json(indent=2),
                        encoding="utf-8",
                    )
                    return parsed, raw, prompt_hash
                except (
                    httpx.HTTPError,
                    KeyError,
                    json.JSONDecodeError,
                    ValueError,
                ) as exc:
                    last_error = exc
                    if attempt < self.config.max_retries:
                        await asyncio.sleep(2**attempt)
        raise RuntimeError("judge failed after bounded retries") from last_error

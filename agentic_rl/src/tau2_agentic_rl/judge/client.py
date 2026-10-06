"""Async OpenAI-compatible client used only for final trajectory judging."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from tau2_agentic_rl.concurrency import api_budget
from tau2_agentic_rl.failures import JudgeServiceFailure
from tau2_agentic_rl.judge.evidence import validate_evidence_turn_ids
from tau2_agentic_rl.judge.prompts import (
    JUDGE_PROMPT_VERSION,
    JUDGE_RUBRIC_VERSION,
    JUDGE_SCHEMA_VERSION,
    build_judge_messages,
)
from tau2_agentic_rl.policy_rules import definitely_active_policy_rules
from tau2_agentic_rl.schemas import JudgeResult, TransferCheck
from tau2_agentic_rl.versions import sha256_json


@dataclass(frozen=True)
class JudgeConfig:
    """External judge endpoint configuration."""

    model: str
    provider: str = "DeepSeek"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    timeout_seconds: float = 120.0
    max_retries: int = 2
    cache_dir: str = "outputs/judge_cache"
    temperature: float = 0.0
    prompt_version: str = JUDGE_PROMPT_VERSION
    rubric_version: str = JUDGE_RUBRIC_VERSION
    schema_version: str = JUDGE_SCHEMA_VERSION
    scorer_code_version: str = "tau2-agentic-rl-scorer-v5-policy-applicability"


class DeepSeekJudge:
    """One isolated, cached judge call per complete trajectory."""

    result_model = JudgeResult
    build_messages = staticmethod(build_judge_messages)

    def __init__(self, config: JudgeConfig):
        if config.model.startswith("FIX_EXACT_"):
            raise ValueError("set an exact judge model ID before running")
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_identity(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Return every field that can change a cached judge decision."""
        return {
            "provider": self.config.provider,
            "base_url": self.config.base_url.rstrip("/"),
            "model_id": self.config.model,
            "messages": messages,
            "schema_version": self.config.schema_version,
            "rubric_version": self.config.rubric_version,
            "prompt_version": self.config.prompt_version,
            "decoding_config": {
                "temperature": self.config.temperature,
                "response_format": {"type": "json_object"},
            },
            "scorer_code_version": self.config.scorer_code_version,
        }

    @staticmethod
    def _atomic_write(path: Path, payload: str) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @staticmethod
    def _validate_requested_criteria(
        result: JudgeResult, inputs: dict[str, Any]
    ) -> None:
        """Require exact unique IDs, then canonicalize decisions by rubric ID."""

        expected_semantic = [
            str(item["criterion_id"]) for item in inputs.get("semantic_checks", [])
        ]
        expected_policy = [
            str(item.get("criterion_id", item.get("rule_id")))
            for item in inputs.get("mandatory_policy_checks", [])
        ]
        if "None" in expected_policy:
            raise ValueError("each mandatory judge check needs criterion_id or rule_id")
        expected_transfer_semantic = [
            str(item["criterion_id"])
            for item in inputs.get("transfer_rule", {}).get("semantic_checks", [])
        ]
        for field, expected in (
            ("semantic_checks", expected_semantic),
            ("transfer_semantic_checks", expected_transfer_semantic),
            ("mandatory_policy_checks", expected_policy),
        ):
            checks = getattr(result, field)
            actual = [check.criterion_id for check in checks]
            if (
                len(set(expected)) != len(expected)
                or len(set(actual)) != len(actual)
                or set(actual) != set(expected)
            ):
                raise ValueError(
                    f"judge {field} criterion IDs do not exactly match the requested rubric: "
                    f"expected={expected}, actual={actual}"
                )
            by_id = {check.criterion_id: check for check in checks}
            setattr(result, field, [by_id[criterion_id] for criterion_id in expected])

        active = definitely_active_policy_rules(
            inputs.get("trajectory", {}), inputs.get("transfer_rule", {})
        )
        for definition, check in zip(inputs.get("mandatory_policy_checks", []),
                                      result.mandatory_policy_checks, strict=True):
            if "applicability" in definition and "applicable" not in check.model_fields_set:
                raise ValueError("new policy rubric requires explicit applicability")
            if not check.applicable and check.criterion_id.rsplit(":policy:", 1)[-1] in active:
                raise ValueError("Judge cannot mark an active policy rule as not applicable")
            if not check.applicable and not check.short_reason.strip():
                raise ValueError("inapplicable policy requires a concrete trigger-absence reason")
            if inputs.get("require_policy_attribution") and not check.passed and not check.violation_assistant_turn_ids:
                raise ValueError("turn-only policy scoring requires explicit violation attribution")
            if not check.passed and (
                not (check.evidence_turn_ids or check.violation_assistant_turn_ids)
                or not check.short_reason.strip()
            ):
                raise ValueError(
                    "failed policy criterion requires evidence and a concrete reason"
                )

        tool_events = inputs.get("trajectory", {}).get("tool_events", [])
        transferred = any(
            item.get("name") == "transfer_to_human_agents" and item.get("success")
            for item in tool_events
        )
        transfer_required = bool(inputs.get("transfer_rule", {}).get("required", False))
        if inputs.get("require_policy_attribution") and transfer_required and not transferred:
            transfer_verdicts = [
                check for check in result.mandatory_policy_checks
                if check.criterion_id.endswith(":policy:transfer_scope_and_message")
            ]
            if not transfer_verdicts or any(check.passed for check in transfer_verdicts):
                # Otherwise score_trajectory rejects the omission after this
                # response is cached, and frozen retries cannot repair it.
                raise ValueError("missing required transfer needs an attributed policy failure")
        expected_transfer_applicable = transferred or transfer_required
        if not transferred:
            # These are environment facts, not model decisions. Keep the raw
            # response for audit, but never discard an otherwise scoreable
            # rollout because the Judge imagined an executed transfer.
            result.transfer_check = TransferCheck(
                applicable=expected_transfer_applicable,
                valid=False,
                short_reason="No successful transfer tool execution.",
            )
        else:
            # A model that said 'not applicable' did not establish validity.
            result.transfer_check.valid = (
                result.transfer_check.applicable and result.transfer_check.valid
            )
            result.transfer_check.applicable = True
        validate_evidence_turn_ids(result, inputs.get("trajectory", {}))

    async def evaluate(self, **inputs: Any) -> tuple[JudgeResult, str, str, str]:
        """Return result, raw response, prompt hash, and full cache-key hash."""
        messages = self.build_messages(**inputs)
        prompt_hash = sha256_json(messages)
        identity = self.cache_identity(messages)
        cache_key = sha256_json(identity)
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            envelope = json.loads(cache_path.read_text(encoding="utf-8"))
            if envelope.get("identity") != identity:
                raise RuntimeError("judge cache identity mismatch")
            parsed = self.result_model.model_validate(envelope["result"])
            raw = str(envelope.get("raw_response", ""))
            self._validate_requested_criteria(parsed, inputs)
            return parsed, raw, prompt_hash, cache_key

        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing {self.config.api_key_env}")
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "response_format": {"type": "json_object"},
        }
        last_error: Exception | None = None
        async with httpx.AsyncClient(
            base_url=self.config.base_url.rstrip("/") + "/",
            timeout=self.config.timeout_seconds,
        ) as client:
            for attempt in range(self.config.max_retries + 1):
                try:
                    async with api_budget().aslot("judge_api"):
                        response = await client.post(
                            "chat/completions",
                            headers={"Authorization": f"Bearer {api_key}"},
                            json=payload,
                        )
                    response.raise_for_status()
                    raw = response.json()["choices"][0]["message"]["content"]
                    parsed = self.result_model.model_validate(json.loads(raw))
                    self._validate_requested_criteria(parsed, inputs)
                    self._atomic_write(
                        cache_path,
                        json.dumps(
                            {
                                "identity": identity,
                                "result": parsed.model_dump(mode="json"),
                                "raw_response": raw,
                            },
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                        ),
                    )
                    return parsed, raw, prompt_hash, cache_key
                except (
                    httpx.HTTPError,
                    KeyError,
                    json.JSONDecodeError,
                    ValueError,
                ) as exc:
                    last_error = exc
                    if isinstance(exc, httpx.HTTPError):
                        from tau2_agentic_rl.slot_recovery import interaction_retryable

                        if not interaction_retryable("model_generation", exc):
                            break
                    if attempt < self.config.max_retries:
                        await asyncio.sleep(2**attempt)
        from tau2_agentic_rl.slot_recovery import interaction_retryable

        retryable = not isinstance(last_error, httpx.HTTPError) or interaction_retryable(
            "model_generation", last_error
        )
        raise JudgeServiceFailure("judge failed after bounded retries", retryable=retryable) from last_error

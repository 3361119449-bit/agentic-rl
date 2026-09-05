"""Validated schemas shared by rollout, reward, and offline rescoring."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

TerminationReason = Literal[
    "agent_stop",
    "user_stop",
    "human_transfer",
    "environment_terminated",
    "budget_exhausted",
    "generation_truncated",
    "hard_turn_limit",
    "infrastructure_failure",
    "max_steps",
    "timeout",
    "too_many_errors",
    "agent_error",
    "user_error",
    "infrastructure_error",
    "context_window_exceeded",
    "unexpected_error",
]

ToolErrorKind = Literal[
    "parse_error",
    "unknown_tool",
    "schema_invalid",
    "model_caused_execution_error",
    "confirmation_required",
    "multiple_tool_calls",
    "mixed_content_and_tool_call",
]


class TokenTurn(BaseModel):
    """Original token-in/token-out data for one actor generation."""

    assistant_turn_index: int = Field(ge=0)
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    output_old_log_probs: list[float]

    @model_validator(mode="after")
    def validate_token_logprob_alignment(self) -> "TokenTurn":
        if len(self.output_token_ids) != len(self.output_old_log_probs):
            raise ValueError("output token and old-log-prob lengths differ")
        return self


class ToolEvent(BaseModel):
    """One parsed actor tool call and its observed execution outcome."""

    event_id: str
    sequence: int = Field(ge=0)
    turn_id: int = Field(ge=0)
    name: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    success: bool = False
    db_effect: bool | None = None
    confirmed_before: bool | None = None
    confirmation_proposal_hash: str | None = None
    confirmation_turn_id: int | None = None
    confirmation_consumed: bool = False
    observation_truncated: bool = False
    error_kind: ToolErrorKind | None = None
    unchanged_retry: bool = False
    no_progress: bool = False
    result: str | None = None


class JudgeCheck(BaseModel):
    """One binary, auditable judge decision."""

    criterion_id: str
    passed: bool
    evidence_turn_ids: list[int] = Field(default_factory=list)
    short_reason: str = ""


class TransferCheck(BaseModel):
    """Judge result for the human-transfer branch."""

    applicable: bool = False
    valid: bool = False
    evidence_turn_ids: list[int] = Field(default_factory=list)
    short_reason: str = ""


class JudgeResult(BaseModel):
    """Strict structured output accepted from the external judge."""

    schema_version: Literal["1.0"] = "1.0"
    semantic_checks: list[JudgeCheck] = Field(default_factory=list)
    transfer_semantic_checks: list[JudgeCheck] = Field(default_factory=list)
    mandatory_policy_checks: list[JudgeCheck] = Field(default_factory=list)
    transfer_check: TransferCheck = Field(default_factory=TransferCheck)


class OfficialScores(BaseModel):
    """Tau2 official reward plus components needed by custom scoring."""

    reward: float = Field(ge=0.0, le=1.0)
    db_applicable: bool = False
    db_score: float | None = Field(default=None, ge=0.0, le=1.0)
    communicate_applicable: bool = False
    communicate_partial: float | None = Field(default=None, ge=0.0, le=1.0)
    communicate_all: bool | None = None
    raw_reward_info: dict[str, Any] = Field(default_factory=dict)


class ComponentScore(BaseModel):
    """A reward component that may be inapplicable to a task."""

    applicable: bool
    value: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_applicability(self) -> "ComponentScore":
        if self.applicable != (self.value is not None):
            raise ValueError("applicable components need a value and vice versa")
        return self


class RequiredActionResult(BaseModel):
    """Required-action matching result."""

    component: ComponentScore
    required_action_ids: list[str] = Field(default_factory=list)
    completed_action_ids: list[str] = Field(default_factory=list)
    missing_action_ids: list[str] = Field(default_factory=list)
    matched_event_ids: list[str] = Field(default_factory=list)


class PolicyCheckResult(BaseModel):
    """Deterministic or judge-backed mandatory policy check."""

    rule_id: str
    applicable: bool
    passed: bool
    evidence_event_ids: list[str] = Field(default_factory=list)
    reason: str = ""


class ProcessPenaltyResult(BaseModel):
    """Deduplicated process-error penalty."""

    penalty: float = Field(ge=0.0, le=1.0)
    process_reward: float = Field(ge=0.0, le=1.0)
    events: list[dict[str, Any]] = Field(default_factory=list)


class RewardResult(BaseModel):
    """Complete custom score while keeping official reward separate."""

    branch: Literal["normal", "human_transfer"]
    train_reward: float = Field(ge=0.0, le=1.0)
    strict_success: float = Field(ge=0.0, le=1.0)
    progress: float = Field(ge=0.0, le=1.0)
    policy_gate: bool
    task_safety_gate: bool = True
    process_penalty: float = Field(ge=0.0, le=1.0)
    components: dict[str, ComponentScore] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)


class TrajectoryRecord(BaseModel):
    """Portable saved trajectory used by training and offline rescoring."""

    schema_version: Literal["1.0"] = "1.0"
    trajectory_id: str
    task_id: str
    split: Literal["train", "internal_dev", "test"]
    policy_version: int = Field(ge=0)
    annotation_version: str
    reward_version: str
    environment_seed: int | None = None
    termination_reason: TerminationReason
    assistant_turns: int = Field(ge=0)
    trajectory_tokens: int = Field(ge=0)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tool_events: list[ToolEvent] = Field(default_factory=list)
    token_turns: list[TokenTurn] = Field(default_factory=list)
    initial_db_hash: str | None = None
    final_db_hash: str | None = None
    target_db_hash: str | None = None
    official_scores: OfficialScores | None = None
    judge_result: JudgeResult | None = None
    custom_reward: RewardResult | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

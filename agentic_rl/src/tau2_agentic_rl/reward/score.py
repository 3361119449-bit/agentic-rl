"""Top-level dual-branch reward composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tau2_agentic_rl.reward.mandatory_policy import (
    evaluate_mandatory_policy,
    policy_gate_passed,
)
from tau2_agentic_rl.reward.normal_branch import (
    normalized_weighted_mean,
    strict_success,
)
from tau2_agentic_rl.reward.process_penalty import (
    ProcessPenaltyConfig,
    compute_process_penalty,
)
from tau2_agentic_rl.reward.required_actions import evaluate_required_actions
from tau2_agentic_rl.reward.transfer_branch import (
    successful_transfer_event,
    transfer_components,
)
from tau2_agentic_rl.schemas import (
    ComponentScore,
    JudgeResult,
    OfficialScores,
    PolicyCheckResult,
    RewardResult,
    ToolEvent,
)


@dataclass(frozen=True)
class RewardConfig:
    """First-version reward coefficients."""

    normal_weights: dict[str, float] = field(
        default_factory=lambda: {
            "db": 0.30,
            "official_communicate": 0.15,
            "required_action": 0.30,
            "judge_semantic": 0.25,
        }
    )
    transfer_weights: dict[str, float] = field(
        default_factory=lambda: {
            "transfer_call": 0.30,
            "pre_transfer_actions": 0.25,
            "transfer_communication": 0.20,
            "transfer_semantic": 0.25,
        }
    )
    progress_coefficient: float = 0.75
    strict_coefficient: float = 0.25
    process: ProcessPenaltyConfig = field(default_factory=ProcessPenaltyConfig)


def _judge_component(judge: JudgeResult) -> ComponentScore:
    checks = judge.semantic_checks
    if not checks:
        return ComponentScore(applicable=False, value=None)
    return ComponentScore(
        applicable=True,
        value=sum(item.passed for item in checks) / len(checks),
    )


def _judge_policy_checks(judge: JudgeResult) -> list[PolicyCheckResult]:
    return [
        PolicyCheckResult(
            rule_id=item.criterion_id,
            applicable=True,
            passed=item.passed,
            reason=item.short_reason,
        )
        for item in judge.mandatory_policy_checks
    ]


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_trajectory(
    *,
    events: list[ToolEvent],
    messages: list[dict[str, Any]],
    assistant_turns: int,
    required_actions: list[dict[str, Any]],
    official: OfficialScores,
    judge: JudgeResult,
    transfer_rule: dict[str, Any] | None = None,
    config: RewardConfig | None = None,
) -> RewardResult:
    """Compute custom training reward while preserving official score separately."""
    config = config or RewardConfig()
    process = compute_process_penalty(events, assistant_turns, config.process)
    policy_checks = evaluate_mandatory_policy(
        events,
        required_actions,
        _judge_policy_checks(judge),
    )
    transfer_event = successful_transfer_event(events)
    if bool((transfer_rule or {}).get("required", False)):
        policy_checks.append(
            PolicyCheckResult(
                rule_id="required_human_transfer_completed",
                applicable=True,
                passed=transfer_event is not None,
                evidence_event_ids=(
                    [transfer_event.event_id] if transfer_event is not None else []
                ),
                reason=(
                    "required human transfer executed successfully"
                    if transfer_event is not None
                    else "task required human transfer but none succeeded"
                ),
            )
        )
    policy_gate = policy_gate_passed(policy_checks)

    if transfer_event is not None:
        valid, components = transfer_components(
            events,
            messages,
            transfer_rule or {},
            judge,
        )
        progress = normalized_weighted_mean(components, config.transfer_weights)
        strict = strict_success(components)
        reward = _clip(
            config.progress_coefficient * progress
            + config.strict_coefficient * strict
            - process.penalty
        )
        if not policy_gate or not valid:
            reward = 0.0
            strict = 0.0
        return RewardResult(
            branch="human_transfer",
            train_reward=reward,
            strict_success=strict,
            progress=progress,
            policy_gate=policy_gate and valid,
            process_penalty=process.penalty,
            components=components,
            details={
                "transfer_valid": valid,
                "policy_checks": [item.model_dump() for item in policy_checks],
                "process_events": process.events,
            },
        )

    action_result = evaluate_required_actions(required_actions, events)
    components = {
        "db": ComponentScore(
            applicable=official.db_applicable and official.db_score is not None,
            value=official.db_score,
        ),
        "official_communicate": ComponentScore(
            applicable=(
                official.communicate_applicable
                and official.communicate_partial is not None
            ),
            value=official.communicate_partial,
        ),
        "required_action": action_result.component,
        "judge_semantic": _judge_component(judge),
    }
    progress = normalized_weighted_mean(components, config.normal_weights)
    strict = strict_success(components)
    reward = _clip(
        config.progress_coefficient * progress
        + config.strict_coefficient * strict
        - process.penalty
    )
    if not policy_gate:
        reward = 0.0
        strict = 0.0
    return RewardResult(
        branch="normal",
        train_reward=reward,
        strict_success=strict,
        progress=progress,
        policy_gate=policy_gate,
        process_penalty=process.penalty,
        components=components,
        details={
            "required_actions": action_result.model_dump(),
            "policy_checks": [item.model_dump() for item in policy_checks],
            "process_events": process.events,
            "tau2_official_reward": official.reward,
        },
    )

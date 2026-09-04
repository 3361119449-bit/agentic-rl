"""Top-level dual-branch reward composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tau2_agentic_rl.reward.mandatory_policy import (
    evaluate_mandatory_policy,
    evaluate_task_safety,
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
    enable_mandatory_policy_gate: bool = True
    enable_task_safety_gate: bool = True
    process: ProcessPenaltyConfig = field(default_factory=ProcessPenaltyConfig)


def build_reward_config(project_config: dict[str, Any]) -> RewardConfig:
    """Map the versioned YAML schema to the runtime reward dataclasses."""
    reward = project_config.get("reward", {})
    rollout = project_config.get("rollout", {})
    defaults = RewardConfig()
    process_defaults = defaults.process
    return RewardConfig(
        normal_weights={
            str(key): float(value)
            for key, value in reward.get(
                "normal_weights", defaults.normal_weights
            ).items()
        },
        transfer_weights={
            str(key): float(value)
            for key, value in reward.get(
                "transfer_weights", defaults.transfer_weights
            ).items()
        },
        progress_coefficient=float(
            reward.get("progress_coefficient", defaults.progress_coefficient)
        ),
        strict_coefficient=float(
            reward.get("strict_success_coefficient", defaults.strict_coefficient)
        ),
        enable_mandatory_policy_gate=bool(
            reward.get("mandatory_policy_gate", defaults.enable_mandatory_policy_gate)
        ),
        enable_task_safety_gate=bool(
            reward.get("task_safety_gate", defaults.enable_task_safety_gate)
        ),
        process=ProcessPenaltyConfig(
            penalties={
                str(key): float(value)
                for key, value in reward.get(
                    "process_penalties", process_defaults.penalties
                ).items()
            },
            cap=float(reward.get("process_penalty_cap", process_defaults.cap)),
            soft_turn_limit=int(
                rollout.get("max_soft_turns", process_defaults.soft_turn_limit)
            ),
            over_turn_cap=float(
                reward.get("over_turn_penalty_cap", process_defaults.over_turn_cap)
            ),
        ),
    )


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
    action_dependencies: list[list[str]] | None = None,
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
    task_safety_check = evaluate_task_safety(events, required_actions)
    task_safety_gate = not task_safety_check.applicable or task_safety_check.passed
    policy_blocks_reward = config.enable_mandatory_policy_gate and not policy_gate
    safety_blocks_reward = config.enable_task_safety_gate and not task_safety_gate

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
        if policy_blocks_reward or safety_blocks_reward or not valid:
            reward = 0.0
            strict = 0.0
        return RewardResult(
            branch="human_transfer",
            train_reward=reward,
            strict_success=strict,
            progress=progress,
            policy_gate=policy_gate and valid,
            task_safety_gate=task_safety_gate,
            process_penalty=process.penalty,
            components=components,
            details={
                "transfer_valid": valid,
                "policy_checks": [item.model_dump() for item in policy_checks],
                "task_safety_check": task_safety_check.model_dump(),
                "process_events": process.events,
            },
        )

    action_result = evaluate_required_actions(
        required_actions, events, action_dependencies
    )
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
    if policy_blocks_reward or safety_blocks_reward:
        reward = 0.0
        strict = 0.0
    return RewardResult(
        branch="normal",
        train_reward=reward,
        strict_success=strict,
        progress=progress,
        policy_gate=policy_gate,
        task_safety_gate=task_safety_gate,
        process_penalty=process.penalty,
        components=components,
        details={
            "required_actions": action_result.model_dump(),
            "policy_checks": [item.model_dump() for item in policy_checks],
            "task_safety_check": task_safety_check.model_dump(),
            "process_events": process.events,
            "tau2_official_reward": official.reward,
        },
    )

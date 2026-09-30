"""Top-level dual-branch reward composition."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any

from tau2_agentic_rl.judge.evidence import validate_evidence_turn_ids
from tau2_agentic_rl.reward.mandatory_policy import (
    evaluate_mandatory_policy,
    evaluate_task_safety,
    policy_gate_passed,
)
from tau2_agentic_rl.reward.normal_branch import (
    normalized_weighted_mean,
    strict_success,
)
from tau2_agentic_rl.reward.policy_credit import build_policy_credit
from tau2_agentic_rl.reward.process_penalty import (
    ProcessPenaltyConfig,
    build_process_credit,
    compute_process_penalty,
)
from tau2_agentic_rl.reward.progress import (
    validate_progress_inputs,
    validate_progress_trace,
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
    ProcessPenaltyResult,
    RewardResult,
    TerminationReason,
    ToolEvent,
)

TRUNCATED_TERMINATIONS = frozenset(
    {
        "budget_exhausted",
        "generation_truncated",
        "hard_turn_limit",
        "max_steps",
        "context_window_exceeded",
    }
)


@dataclass(frozen=True)
class RewardConfig:
    """First-version reward coefficients."""

    mode: str = "legacy"
    progress_scale: float = 0.5
    credit_version: str = "procredit-turn-v1"
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
    truncation_multiplier: float = 0.75
    enable_mandatory_policy_gate: bool = True
    enable_task_safety_gate: bool = True
    process: ProcessPenaltyConfig = field(default_factory=ProcessPenaltyConfig)

    def __post_init__(self) -> None:
        if self.credit_version not in {"procredit-turn-v1", "procredit-turn-v2", "procredit-turn-v3", "procredit-turn-v4"}:
            raise ValueError("unknown credit version")
        local_only = self.credit_version == "procredit-turn-v4"
        if local_only != (self.mode == "turn_local_v1"):
            raise ValueError("turn_local_v1 requires ProCredit v4")
        if self.credit_version != "procredit-turn-v1" and self.mode not in {"strict_progress_v1", "turn_local_v1"}:
            raise ValueError("policy-local credit requires strict terminal reward")
        if self.mode not in {"legacy", "strict_progress_v1", "turn_local_v1"}:
            raise ValueError("unknown reward mode")
        if self.mode in {"strict_progress_v1", "turn_local_v1"} and (
            self.progress_scale != 0.5
            or self.truncation_multiplier != 0.75
            or self.enable_mandatory_policy_gate == local_only
            or self.enable_task_safety_gate == local_only
            or (not local_only and self.process.cap != 0.20)
            or (local_only and (self.process.cap is not None or self.process.over_turn_cap is not None))
        ):
            raise ValueError("ProCredit requires c=.5, truncation=.75 and version-specific gates/process caps")
        if not isfinite(self.truncation_multiplier) or not (
            0.0 <= self.truncation_multiplier <= 1.0
        ):
            raise ValueError("truncation_multiplier must be finite and within [0, 1]")


def build_reward_config(project_config: dict[str, Any]) -> RewardConfig:
    """Map the versioned YAML schema to the runtime reward dataclasses."""
    reward = project_config.get("reward", {})
    local_only = project_config.get("credit", {}).get("version") == "procredit-turn-v4"
    if local_only and {"process_penalty_cap", "over_turn_penalty_cap"} & reward.keys():
        raise ValueError("v4 process penalties are local and cannot use trajectory caps")
    if reward.get("mode") in {"strict_progress_v1", "turn_local_v1"}:
        legacy = {
            "normal_weights", "transfer_weights", "progress_coefficient",
            "strict_success_coefficient",
        } & reward.keys()
        if legacy:
            raise ValueError(f"new reward mode cannot include legacy weights: {legacy}")
    rollout = project_config.get("rollout", {})
    defaults = RewardConfig()
    process_defaults = defaults.process
    return RewardConfig(
        mode=reward.get("mode", "legacy"),
        credit_version=project_config.get("credit", {}).get("version", "procredit-turn-v1"),
        progress_scale=float(reward.get("progress_scale", 0.5)),
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
        truncation_multiplier=float(
            reward.get("truncation_multiplier", defaults.truncation_multiplier)
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
            cap=None if local_only else float(reward.get("process_penalty_cap", process_defaults.cap)),
            soft_turn_limit=int(
                rollout.get("max_soft_turns", process_defaults.soft_turn_limit)
            ),
            over_turn_cap=None if local_only else float(
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
            applicable=item.applicable,
            passed=item.passed,
            reason=item.short_reason,
        )
        for item in judge.mandatory_policy_checks
    ]


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _score_legacy(
    *,
    events: list[ToolEvent],
    messages: list[dict[str, Any]],
    assistant_turns: int,
    required_actions: list[dict[str, Any]],
    official: OfficialScores,
    judge: JudgeResult,
    transfer_rule: dict[str, Any] | None = None,
    action_dependencies: list[list[str]] | None = None,
    termination_reason: TerminationReason | None = None,
    config: RewardConfig | None = None,
) -> RewardResult:
    """Compute custom training reward while preserving official score separately."""
    config = config or RewardConfig()
    validate_evidence_turn_ids(
        judge,
        {"messages": messages, "tool_events": [event.model_dump() for event in events]},
    )
    truncated = termination_reason in TRUNCATED_TERMINATIONS
    multiplier = config.truncation_multiplier if truncated else 1.0
    process = (
        ProcessPenaltyResult(penalty=0, process_reward=1, events=[])
        if config.credit_version == "procredit-turn-v4"
        else compute_process_penalty(events, assistant_turns, config.process)
    )
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

    local_only = config.credit_version == "procredit-turn-v4"
    transfer_target = bool((transfer_rule or {}).get("required"))
    transfer_valid, transfer_scores = None, None
    if transfer_event is not None or (local_only and transfer_target):
        transfer_valid, transfer_scores = transfer_components(
            events,
            messages,
            transfer_rule or {},
            judge,
        )
    if (local_only and transfer_target) or (not local_only and transfer_event is not None):
        valid, components = transfer_valid, transfer_scores
        progress = normalized_weighted_mean(components, config.transfer_weights)
        strict = strict_success(components)
        reward = _clip(
            config.progress_coefficient * progress
            + config.strict_coefficient * strict
            - process.penalty
        )
        if policy_blocks_reward or safety_blocks_reward or (
            not valid and config.credit_version != "procredit-turn-v4"
        ):
            reward = 0.0
            strict = 0.0
        return RewardResult(
            branch="human_transfer",
            train_reward=reward * multiplier,
            strict_success=strict,
            progress=progress,
            policy_gate=policy_gate and valid,
            task_safety_gate=task_safety_gate,
            process_penalty=process.penalty,
            components=components,
            details={
                "termination_reason": termination_reason,
                "trajectory_truncated": truncated,
                "truncation_multiplier": multiplier,
                "reward_before_truncation": reward,
                "transfer_valid": valid,
                "policy_checks": [item.model_dump() for item in policy_checks],
                "task_safety_check": task_safety_check.model_dump(),
                "process_events": process.events,
            },
        )

    if local_only and transfer_event is not None:
        policy_gate = policy_gate and bool(transfer_valid)
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
        train_reward=reward * multiplier,
        strict_success=strict,
        progress=progress,
        policy_gate=policy_gate,
        task_safety_gate=task_safety_gate,
        process_penalty=process.penalty,
        components=components,
        details={
            "termination_reason": termination_reason,
            "trajectory_truncated": truncated,
            "truncation_multiplier": multiplier,
            "reward_before_truncation": reward,
            "required_actions": action_result.model_dump(),
            **({"transfer_valid": transfer_valid} if local_only and transfer_event is not None else {}),
            "policy_checks": [item.model_dump() for item in policy_checks],
            "task_safety_check": task_safety_check.model_dump(),
            "process_events": process.events,
            "tau2_official_reward": official.reward,
        },
    )


def score_trajectory(
    *, config: RewardConfig | None = None,
    progress_trace: dict[str, Any] | None = None, **kwargs: Any,
) -> RewardResult:
    """Preserve strict/gate semantics and version only the training score."""
    config = config or RewardConfig()
    result = _score_legacy(config=config, **kwargs)
    if config.mode == "legacy":
        return result
    if progress_trace is None:
        raise ValueError("ProCredit reward requires frozen progress")
    if config.credit_version in {"procredit-turn-v3", "procredit-turn-v4"} and progress_trace.get("version") != "progress-v2":
        raise ValueError("ProCredit v3 requires progress-v2")
    phi = validate_progress_trace(progress_trace, turns=kwargs["assistant_turns"])
    validate_progress_inputs(
        progress_trace, required_actions=kwargs["required_actions"],
        dependencies=kwargs.get("action_dependencies") or [],
        transfer_rule=kwargs.get("transfer_rule") or {},
    )
    valid = result.policy_gate and result.task_safety_gate
    local_only = config.credit_version == "procredit-turn-v4"
    multiplier = result.details["truncation_multiplier"]
    task_score = result.strict_success + config.progress_scale * phi[-1]
    raw_score = multiplier * task_score
    if not local_only:
        raw_score -= result.process_penalty
    details = {
        **result.details,
        "task_score": task_score,
        "score_before_floor": raw_score,
        "score_floor_applied": (valid or local_only) and raw_score < 0,
        "progress_version": progress_trace["version"],
        "checkset_fingerprint": progress_trace["checkset_fingerprint"],
    }
    # The legacy pre-truncation quantity is not the new reward's intermediate.
    details.pop("reward_before_truncation", None)
    scored = RewardResult.model_validate({
        **result.model_dump(),
        "reward_mode": config.mode,
        "train_reward": max(0.0, raw_score) if valid or local_only else 0.0,
        "strict_success": result.strict_success if valid else 0.0,
        "progress": phi[-1],
        "details": details,
    })
    if config.credit_version != "procredit-turn-v1":
        scored.details["credit_version"] = config.credit_version
        scored.details["policy_credit"] = build_policy_credit(
            reward=scored, judge=kwargs["judge"], events=kwargs["events"],
            turns=kwargs["assistant_turns"],
        )
    if local_only:
        policy = scored.details["policy_credit"]
        if not policy["attribution_complete"]:
            raise ValueError("v4 requires complete policy attribution; retry this frozen Judge input")
        scored.details["task_completion"] = result.strict_success
        scored.details["policy_gate_applied"] = False
        scored.details["policy_turn_rewards"] = [
            -1.0 if t in policy["violating_turns"] else 0.0
            for t in range(kwargs["assistant_turns"])
        ]
    if config.credit_version in {"procredit-turn-v3", "procredit-turn-v4"}:
        scored.details["process_config"] = asdict(config.process)
        scored.details["process_credit"] = build_process_credit(
            kwargs["events"], kwargs["assistant_turns"], config.process, local_only=local_only,
        )
        if local_only:
            scored.details["process_events"] = scored.details["process_credit"]["events"]
            scored.details["process_penalty_placement"] = "turn_only"
    return scored

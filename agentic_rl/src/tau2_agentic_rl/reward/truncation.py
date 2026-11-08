"""Versioned truncation decisions shared by training, retry and evaluation."""
from tau2_agentic_rl.schemas import ComponentScore, RewardResult

ZERO_TRUNCATION = "zero-trajectory-v1"
TRUNCATED_TERMINATIONS = frozenset({
    "budget_exhausted", "generation_truncated", "hard_turn_limit",
    "max_steps", "context_window_exceeded",
})


def zero_truncated(project, termination_reason):
    policy = project.get("reward", {}).get("truncation_policy")
    if policy not in (None, ZERO_TRUNCATION):
        raise ValueError("unknown truncation policy")
    return policy == ZERO_TRUNCATION and termination_reason in TRUNCATED_TERMINATIONS


def deterministic_zero_reward(*, termination_reason, project, official=None):
    if not zero_truncated(project, termination_reason):
        raise ValueError("deterministic zero requires explicit policy and actual truncation")
    reward = project.get("reward", {})
    if reward.get("truncation_multiplier") != 0:
        raise ValueError("zero truncation policy requires multiplier0")
    components = {}
    if official is not None:
        components["db"] = ComponentScore(
            applicable=official.db_applicable and official.db_score is not None,
            value=official.db_score)
        components["official_communicate"] = ComponentScore(
            applicable=official.communicate_applicable and official.communicate_partial is not None,
            value=official.communicate_partial)
    return RewardResult(
        branch="normal", reward_mode=reward.get("mode", "legacy"),
        train_reward=0, strict_success=0, progress=0, policy_gate=False,
        task_safety_gate=False, process_penalty=0, components=components,
        details={
            "termination_reason": termination_reason, "trajectory_truncated": True,
            "truncation_policy": ZERO_TRUNCATION, "truncation_zero_applied": True,
            "truncation_multiplier": 0.0, "task_completion": 0.0,
            "effective_progress_reward": 0.0,
            "scoring_source": "deterministic_truncation_no_judge",
            "judge_skipped": True,
            "credit_version": project.get("credit", {}).get("version", "procredit-turn-v1"),
            "policy_checks_not_evaluated": True,
        })


def truncation_metadata(project, reason, *, split, evaluation, slot_context=None):
    context = slot_context or {}
    occurred = reason in TRUNCATED_TERMINATIONS
    limit = project.get("slot_recovery", {}).get("max_truncation_resamples", 0)
    used = context.get("truncation_replacements", 0)
    train = split == "train" and not evaluation
    return {
        "occurred": occurred, "reason": reason if occurred else None,
        "policy": project.get("reward", {}).get("truncation_policy"),
        "resamples_used": used, "resample_limit": limit if train else 0,
        "will_resample": occurred and train and used < limit,
        "resamples_exhausted": occurred and train and limit > 0 and used >= limit,
    }

"""Retry scoring a frozen interaction; never generate a replacement trajectory."""

from tau2_agentic_rl.reward.score import build_reward_config, score_trajectory
from tau2_agentic_rl.versions import sha256_json

SCORING_FAILURES = {"judge", "reward_scoring"}


def scoring_pending(row):
    return (
        row.get("official_scores") is not None
        and row.get("custom_reward") is None
        and row.get("metadata", {}).get("failure_phase") in SCORING_FAILURES
    )


async def retry_scoring(record, judge, store):
    """Atomically update the same ID/slot, preserving all interaction evidence."""
    if not scoring_pending(record.model_dump()):
        raise ValueError("record is not a scoring-only failure")
    inputs = record.scoring_inputs
    if not inputs or sha256_json(inputs) != record.metadata.get(
        "scoring_inputs_sha256"
    ):
        raise ValueError(
            "missing or changed frozen scoring inputs; do not reroll this slot"
        )
    trajectory = inputs["judge"]["trajectory"]
    if (
        record.environment_transcript != trajectory["messages"]
        or [event.model_dump() for event in record.tool_events]
        != trajectory["tool_events"]
        or record.termination_reason != trajectory["termination_reason"]
        or record.official_scores.model_dump() != inputs["official_scores"]
    ):
        raise ValueError("interaction differs from frozen scoring inputs")
    attempts = record.metadata.setdefault("scoring_retries", [])
    attempt = {"attempt": len(attempts) + 1, "phase": "judge", "success": False}
    try:
        result, raw, prompt_hash, cache_key = await judge.evaluate(**inputs["judge"])
        attempt["phase"] = "reward_scoring"
        reward = score_trajectory(
            events=record.tool_events,
            messages=record.environment_transcript,
            assistant_turns=record.assistant_turns,
            required_actions=inputs["required_actions"],
            official=record.official_scores,
            judge=result,
            transfer_rule=inputs["judge"]["transfer_rule"],
            action_dependencies=inputs["action_dependencies"],
            config=build_reward_config(inputs["reward_project_config"]),
        )
        record.judge_result = result
        record.custom_reward = reward
        record.metadata.update(
            judge_raw=raw, judge_prompt_hash=prompt_hash, judge_cache_key=cache_key
        )
        for key in ("failure_phase", "failure_type", "failure_message"):
            record.metadata[key] = None
        attempt["success"] = True
    except Exception as exc:
        attempt["error"] = f"{type(exc).__name__}: {exc}"
        record.metadata.update(
            failure_phase=attempt["phase"],
            failure_type=type(exc).__name__,
            failure_message=str(exc),
        )
    attempts.append(attempt)
    store.save(record)
    return attempt["success"]

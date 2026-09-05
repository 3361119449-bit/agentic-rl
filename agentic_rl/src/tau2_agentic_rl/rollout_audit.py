"""Keep actor context, delivered interaction, and cleanup semantics separate."""

from copy import deepcopy

from tau2_agentic_rl.reward.official_tau2 import parse_official_reward_info

EXTERNAL_STOPS = frozenset(
    {
        "budget_exhausted",
        "hard_turn_limit",
        "generation_truncated",
    }
)


def snapshot_transcript(environment):
    """Snapshot the official interaction BEFORE artificial cleanup, without cropping."""
    messages = deepcopy(environment.full_trajectory())
    for index, message in enumerate(messages):
        message["turn_idx"] = index
    return messages


def audited_official_scores(reward, payload, termination_reason):
    """Only override official success; preserve completed-state reward components."""
    return parse_official_reward_info(
        0.0 if termination_reason in EXTERNAL_STOPS else reward, payload
    )

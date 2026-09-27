"""Pure, shared ProCredit calculation for filtering, training and offline audits."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CreditConfig:
    progress_scale: float = 0.5
    turn_coefficient: float = 1.0
    epsilon: float = 1e-6
    signal_tolerance: float = 1e-8

    def __post_init__(self):
        if self.progress_scale != 0.5 or self.turn_coefficient not in (0.0, 1.0):
            raise ValueError("ProCredit v1 requires c=0.5 and turn coefficient 0 or 1")
        if self.epsilon != 1e-6 or self.signal_tolerance != 1e-8:
            raise ValueError("ProCredit v1 requires epsilon=1e-6, tolerance=1e-8")

    @classmethod
    def from_project(cls, project: dict) -> CreditConfig:
        credit = project.get("credit", {})
        return cls(
            progress_scale=project.get("reward", {}).get("progress_scale", 0.5),
            turn_coefficient=credit.get("turn_coefficient", 1.0),
            epsilon=credit.get("epsilon", 1e-6),
            signal_tolerance=project.get("dynamic_sampling", {}).get(
                "signal_tolerance", 1e-8
            ),
        )


def is_procredit(project: dict) -> bool:
    return project.get("reward", {}).get("mode") == "strict_progress_v1"


def _number(value: Any, low: float, high: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(f"invalid {name}: expected finite value in [{low}, {high}]")
    return float(value)


def validate_phi(phi: list, turns: int | None = None) -> list[float]:
    if not isinstance(phi, list) or not phi:
        raise ValueError("missing progress phi")
    if turns is not None and len(phi) != turns + 1:
        raise ValueError("progress length differs from assistant turns")
    values = [_number(value, 0.0, 1.0, "progress phi") for value in phi]
    if values[0] != 0:
        raise ValueError("progress must start at zero")
    return values


def compute_group_credit(rows: list[dict], config: CreditConfig | None = None) -> dict:
    """Compute a complete sampling group's advantages, without token reweighting.

    The queue caller enforces the configured group size and uid membership.
    This function supports smaller hand-checkable groups for audits.
    """
    config = config or CreditConfig()
    if len(rows) < 2:
        raise ValueError("credit requires a complete group with at least two members")
    identity_fields = (
        "task_id",
        "policy_version",
        "checkset_fingerprint",
        "initial_state_fingerprint",
    )
    identities, ids, validated = [], set(), []
    for row in rows:
        trajectory_id = row.get("trajectory_id")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError("missing trajectory identity")
        if trajectory_id in ids:
            raise ValueError("duplicate trajectory in credit group")
        ids.add(trajectory_id)
        if any(row.get(field) is None for field in identity_fields):
            raise ValueError("missing group identity")
        identities.append(tuple(row[field] for field in identity_fields))
        score = _number(row["score"], 0, 1.5, "score")
        terminal = _number(row["terminal_success"], 0, 1, "terminal_success")
        if terminal not in (0, 1) or type(row["valid"]) is not bool:
            raise ValueError("success and gate must be binary")
        if not row["valid"] and score != 0:
            raise ValueError("invalid trajectory must have zero score")
        multiplier = _number(row["multiplier"], 0.75, 1, "multiplier")
        if multiplier not in (0.75, 1):
            raise ValueError("unsupported truncation multiplier")
        phi = validate_phi(row["phi"])
        mapping = row["response_turn_ids"]
        if not isinstance(mapping, list) or not mapping:
            raise ValueError("missing token-to-turn mapping")
        turns = len(phi) - 1
        if any(type(t) is not int or t < -1 or t >= turns for t in mapping):
            raise ValueError("invalid token-to-turn mapping")
        present = set(mapping) - {-1}
        if not present:
            raise ValueError("trajectory contains no policy tokens")
        returns = [
            multiplier * (terminal + config.progress_scale * (phi[-1] - before))
            for before in phi[:-1]
        ]
        validated.append((score, mapping, present, returns))
    if any(identity != identities[0] for identity in identities):
        raise ValueError("mixed group identity")
    scores = [item[0] for item in validated]
    mean = math.fsum(scores) / len(scores)
    std = math.sqrt(math.fsum((score - mean) ** 2 for score in scores) / len(scores))
    valid_returns = [
        returns[t]
        for row, (_, _, present, returns) in zip(rows, validated, strict=True)
        if row["valid"]
        for t in sorted(present)
    ]
    turn_mean = math.fsum(valid_returns) / len(valid_returns) if valid_returns else 0.0
    results = []
    for row, (score, mapping, present, returns) in zip(rows, validated, strict=True):
        trajectory_advantage = (score - mean) / (std + config.epsilon)
        turn_advantages = [
            value - turn_mean if row["valid"] and t in present else 0.0
            for t, value in enumerate(returns)
        ]
        tokens = [
            trajectory_advantage + config.turn_coefficient * turn_advantages[t]
            if t >= 0
            else 0.0
            for t in mapping
        ]
        results.append(
            {
                "trajectory_id": row["trajectory_id"],
                "trajectory_advantage": trajectory_advantage,
                "turn_returns": returns,
                "turn_advantages": turn_advantages,
                "token_advantages": tokens,
            }
        )
    return {
        "version": "procredit-turn-v1",
        "config": asdict(config),
        "score_mean": mean,
        "score_std": std,
        "turn_mean": turn_mean,
        "valid_turns": len(valid_returns),
        "has_signal": any(
            abs(value) > config.signal_tolerance
            for item in results
            for value in item["token_advantages"]
        ),
        "trajectories": results,
    }

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
    version: str = "procredit-turn-v1"
    violation_penalty: float = 1.0

    def __post_init__(self):
        if self.version not in {"procredit-turn-v1", "procredit-turn-v2"}:
            raise ValueError("unknown ProCredit version")
        if type(self.violation_penalty) is bool or self.violation_penalty != 1.0:
            raise ValueError("policy-local credit requires violation penalty 1.0")
        if self.progress_scale != 0.5 or self.turn_coefficient not in (0.0, 1.0):
            raise ValueError("ProCredit v1 requires c=0.5 and turn coefficient 0 or 1")
        if self.epsilon != 1e-6 or self.signal_tolerance != 1e-8:
            raise ValueError("ProCredit v1 requires epsilon=1e-6, tolerance=1e-8")

    @classmethod
    def from_project(cls, project: dict) -> CreditConfig:
        credit = project.get("credit", {})
        return cls(
            version=credit.get("version", "procredit-turn-v1"),
            violation_penalty=credit.get("violation_penalty", 1.0),
            progress_scale=project.get("reward", {}).get("progress_scale", 0.5),
            turn_coefficient=credit.get("turn_coefficient", 1.0),
            epsilon=credit.get("epsilon", 1e-6),
            signal_tolerance=project.get("dynamic_sampling", {}).get(
                "signal_tolerance", 1e-8
            ),
        )

    def audit_config(self) -> dict:
        values = asdict(self)
        if self.version == "procredit-turn-v1":
            # Preserve existing audit fingerprints and exact offline replay.
            values.pop("version")
            values.pop("violation_penalty")
        return values


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


def _policy_attribution(row: dict, turns: int) -> tuple[set[int], bool]:
    policy = row.get("policy_credit")
    if not isinstance(policy, dict) or policy.get("version") != "policy-credit-v1":
        raise ValueError("v2 requires explicit policy attribution")
    bad, complete = policy.get("violating_turns"), policy.get("attribution_complete")
    unresolved = policy.get("unresolved_checks")
    if (
        not isinstance(bad, list)
        or any(type(t) is not int or not 0 <= t < turns for t in bad)
        or len(set(bad)) != len(bad)
        or type(complete) is not bool
        or not isinstance(unresolved, list)
        or complete != (not unresolved)
        or (row["valid"] and (bad or not complete))
        or (not row["valid"] and complete and not bad)
    ):
        raise ValueError("invalid policy attribution")
    return set(bad), complete


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
    identities, ids, validated, policies = [], set(), [], []
    for row in rows:
        if config.version == "procredit-turn-v1" and "policy_credit" in row:
            raise ValueError("v2 policy credit cannot silently fall back to v1")
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
        if config.version == "procredit-turn-v2":
            bad, complete = _policy_attribution(row, turns)
            if not row["valid"] and terminal != 0:
                raise ValueError("policy-invalid terminal success must remain zero")
            deltas = [
                min(after - before, 0.0) if t in bad else after - before
                for t, (before, after) in enumerate(zip(phi[:-1], phi[1:], strict=True))
            ]
            returns = [
                multiplier * (terminal + config.progress_scale * math.fsum(deltas[t:]))
                if complete and t not in bad else 0.0
                for t in range(turns)
            ]
            policies.append((bad, complete))
        else:
            policies.append((set(), row["valid"]))
        validated.append((score, mapping, present, returns))
    if any(identity != identities[0] for identity in identities):
        raise ValueError("mixed group identity")
    scores = [item[0] for item in validated]
    mean = math.fsum(scores) / len(scores)
    std = math.sqrt(math.fsum((score - mean) ** 2 for score in scores) / len(scores))
    valid_returns = [
        returns[t]
        for (_, complete), (_, _, present, returns) in zip(policies, validated, strict=True)
        if complete
        for t in sorted(present)
    ]
    turn_mean = math.fsum(valid_returns) / len(valid_returns) if valid_returns else 0.0
    results = []
    for row, (score, mapping, present, returns), (bad, complete) in zip(
        rows, validated, policies, strict=True
    ):
        trajectory_advantage = (score - mean) / (std + config.epsilon)
        turn_advantages = [
            value - turn_mean if complete and t in present else 0.0
            for t, value in enumerate(returns)
        ]
        for t in bad & present:
            # Keep penalty local and outside centering. Even an all-bad group
            # must have negative feedback; centering the penalty would erase it.
            turn_advantages[t] = min(turn_advantages[t], 0.0) - config.violation_penalty
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
    result = {
        "version": config.version,
        "config": config.audit_config(),
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
    if config.version == "procredit-turn-v2":
        result["policy_violation_turns"] = sum(len(bad) for bad, _ in policies)
        result["unresolved_trajectories"] = sum(not complete for _, complete in policies)
    return result

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
    db_subgoal_credit: str | None = None
    post_completion_credit: str | None = None

    def __post_init__(self):
        if self.post_completion_credit is not None and (
            self.post_completion_credit != "post_completion_regression_v1" or self.version != "procredit-turn-v4"
        ):
            raise ValueError("post_completion_regression_v1 requires ProCredit v4")
        if self.db_subgoal_credit is not None and (
            self.db_subgoal_credit != "db_subgoal_local_v1" or self.version != "procredit-turn-v4"
        ):
            raise ValueError("db_subgoal_local_v1 requires ProCredit v4")
        if self.version not in {"procredit-turn-v1", "procredit-turn-v2", "procredit-turn-v3", "procredit-turn-v4"}:
            raise ValueError("unknown ProCredit version")
        if self.version == "procredit-turn-v4" and self.turn_coefficient != 1.0:
            raise ValueError("v4 requires mandatory local policy punishment")
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
            post_completion_credit=credit.get("post_completion_credit"),
            db_subgoal_credit=credit.get("db_subgoal_credit"),
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
        if self.post_completion_credit is None:
            values.pop("post_completion_credit")
        if self.db_subgoal_credit is None:
            values.pop("db_subgoal_credit")
        if self.version == "procredit-turn-v1":
            # Preserve existing audit fingerprints and exact offline replay.
            values.pop("version")
            values.pop("violation_penalty")
        return values


def is_procredit(project: dict) -> bool:
    return project.get("reward", {}).get("mode") in {"strict_progress_v1", "turn_local_v1"}


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
    if len(rows) < (1 if config.version == "procredit-turn-v4" else 2):
        raise ValueError("credit requires genuine group members (legacy: at least two)")
    identity_fields = (
        "task_id",
        "policy_version",
        "checkset_fingerprint",
        "initial_state_fingerprint",
    )
    identities, ids, validated, policies, process_costs = [], set(), [], [], []
    for row in rows:
        local_only = config.version == "procredit-turn-v4"
        if (local_only or "credit_version" in row) and row.get("credit_version") != config.version:
            raise ValueError("credit row version differs from algorithm")
        if config.version not in {"procredit-turn-v3", "procredit-turn-v4"} and "process_credit" in row:
            raise ValueError("v3 process credit cannot silently fall back to an older algorithm")
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
        if not local_only and not row["valid"] and score != 0:
            raise ValueError("invalid trajectory must have zero score")
        multiplier = _number(row["multiplier"], 0.75, 1, "multiplier")
        if multiplier not in (0.75, 1):
            raise ValueError("unsupported truncation multiplier")
        phi = validate_phi(row["phi"])
        mapping = row["response_turn_ids"]
        if not isinstance(mapping, list) or not mapping:
            raise ValueError("missing token-to-turn mapping")
        turns = len(phi) - 1
        costs = [0.0] * turns
        if config.version in {"procredit-turn-v3", "procredit-turn-v4"}:
            process = row.get("process_credit")
            process_version = "process-credit-v2" if local_only else "process-credit-v1"
            if not isinstance(process, dict) or process.get("version") != process_version:
                raise ValueError(f"{config.version} requires {process_version}")
            values = process.get("turn_costs")
            if not isinstance(values, list) or len(values) != turns:
                raise ValueError("process costs differ from actor turns")
            costs = [_number(value, 0, math.inf if local_only else .2, "process cost") for value in values]
            if not local_only and math.fsum(costs) > .2 + 1e-12:
                raise ValueError("process costs exceed trajectory cap")
        process_costs.append(costs)
        if any(type(t) is not int or t < -1 or t >= turns for t in mapping):
            raise ValueError("invalid token-to-turn mapping")
        present = set(mapping) - {-1}
        if not present:
            raise ValueError("trajectory contains no policy tokens")
        returns = [
            multiplier * (terminal + config.progress_scale * (phi[-1] - before))
            for before in phi[:-1]
        ]
        if config.version != "procredit-turn-v1":
            bad, complete = _policy_attribution(row, turns)
            if not local_only and not row["valid"] and terminal != 0:
                raise ValueError("policy-invalid terminal success must remain zero")
            if local_only and not complete:
                raise ValueError("v4 requires complete policy attribution")
            if local_only and bad - present:
                raise ValueError("policy violation turn has no trainable tokens")
            if not local_only:
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
    for row, (score, mapping, present, returns), (bad, complete), costs in zip(
        rows, validated, policies, process_costs, strict=True
    ):
        trajectory_advantage = (score - mean) / (std + config.epsilon)
        turn_advantages = [
            value - turn_mean if complete and t in present else 0.0
            for t, value in enumerate(returns)
        ]
        regression_turns = set()
        completion_floor_turn = None
        if config.db_subgoal_credit is not None:
            from tau2_agentic_rl.reward.db_subgoals import validate_db_subgoal_trace

            local = validate_db_subgoal_trace(
                row.get("subgoal_trace"), turns=len(returns), violating_turns=sorted(bad),
            )
            for t in present:
                turn_advantages[t] += local[t]
        if config.post_completion_credit is not None:
            from tau2_agentic_rl.reward.post_completion import (
                validate_post_completion_trace,
            )

            completion_bonus, regression_turns = validate_post_completion_trace(
                row.get("completion_trace"), turns=len(returns), violating_turns=sorted(bad),
                subgoal_trace=row.get("subgoal_trace") if config.db_subgoal_credit is not None else None,
            )
            if regression_turns - present:
                raise ValueError("completion regression turn has no trainable tokens")
            trace = row["completion_trace"]
            if trace["positive_floor_turn"] is not None:
                completion_floor_turn = trace["positive_floor_turn"] - 1
                if completion_floor_turn not in present:
                    raise ValueError("first completion turn has no trainable tokens")
            for t in present:
                turn_advantages[t] += completion_bonus[t]
        for t in present:
            if config.version == "procredit-turn-v4":
                # Costs do not affect any task return or group baseline. Count
                # policy once per turn and add the independent process cost.
                turn_advantages[t] -= (config.violation_penalty if t in bad else 0.0) + costs[t]
                continue
            # Keep penalty local and outside centering. Even an all-bad group
            # must have negative feedback; centering the penalty would erase it.
            penalty = max(config.violation_penalty if t in bad else 0.0, costs[t])
            if penalty:
                turn_advantages[t] = min(turn_advantages[t], 0.0) - penalty
        tokens = [
            trajectory_advantage + config.turn_coefficient * turn_advantages[t]
            if t >= 0
            else 0.0
            for t in mapping
        ]
        if config.post_completion_credit is not None:
            completion_floors = [0.0] * len(returns)
            completion_floor_adjustments = [0.0] * len(returns)
            if completion_floor_turn is not None:
                t = completion_floor_turn
                floor = max(
                    trace["min_positive_advantage"], trace["positive_margin"] - costs[t]
                )
                completion_floors[t] = floor
                # A final sign restriction, not a second additive goal reward.
                # Its correction can exceed the shared 0.5 reward budget when
                # compensating a negative trajectory/group advantage.
                prior = next(
                    value for turn, value in zip(mapping, tokens, strict=True) if turn == t
                )
                completion_floor_adjustments[t] = max(0.0, floor - prior)
                tokens = [
                    max(value, floor) if turn == t else value
                    for turn, value in zip(mapping, tokens, strict=True)
                ]
        if config.version == "procredit-turn-v4":
            # Enforce the sign AFTER the trajectory and local terms combine.
            # A high task score must never reward the violating generation.
            tokens = [
                min(value, -config.violation_penalty - costs[turn]) if turn in bad else value
                for turn, value in zip(mapping, tokens, strict=True)
            ]
        if regression_turns:
            # A sign ceiling, not another additive policy-sized punishment.
            # Existing policy and process penalties retain their exact values.
            tokens = [
                min(value, -1.0 - costs[turn]) if turn in regression_turns else value
                for turn, value in zip(mapping, tokens, strict=True)
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
        if config.version == "procredit-turn-v4":
            results[-1]["policy_turn_rewards"] = [
                -config.violation_penalty if t in bad else 0.0 for t in range(len(returns))
            ]
        if config.db_subgoal_credit is not None:
            results[-1]["db_subgoal_turn_rewards"] = list(local)
        if config.post_completion_credit is not None:
            results[-1]["completion_turn_bonus"] = list(completion_bonus)
            results[-1]["completion_turn_floor"] = completion_floors
            results[-1]["completion_turn_floor_adjustments"] = completion_floor_adjustments
            results[-1]["completion_regression_turns"] = sorted(regression_turns)
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
    if config.version != "procredit-turn-v1":
        result["policy_violation_turns"] = sum(len(bad) for bad, _ in policies)
        result["unresolved_trajectories"] = sum(not complete for _, complete in policies)
    if config.version in {"procredit-turn-v3", "procredit-turn-v4"}:
        result["process_cost_turns"] = sum(value > 0 for costs in process_costs for value in costs)
    return result

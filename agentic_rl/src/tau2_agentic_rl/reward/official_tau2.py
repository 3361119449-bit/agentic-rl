"""Parse Tau2 RewardInfo JSON without conflating it with training reward."""

from __future__ import annotations

import json
from typing import Any

from tau2_agentic_rl.schemas import OfficialScores


def _as_dict(payload: str | dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, str):
        parsed = json.loads(payload)
        return parsed if isinstance(parsed, dict) else {}
    return payload


def parse_official_reward_info(
    official_reward: float,
    reward_info: str | dict[str, Any] | None,
) -> OfficialScores:
    """Extract official DB and per-item COMMUNICATE results."""
    info = _as_dict(reward_info)
    breakdown = {
        str(key).upper(): value
        for key, value in (info.get("reward_breakdown") or {}).items()
    }
    basis = {str(item).upper() for item in (info.get("reward_basis") or [])}

    db_check = info.get("db_check") or {}
    db_value = db_check.get("db_reward", breakdown.get("DB"))
    db_applicable = "DB" in basis or bool(db_check) or db_value is not None
    checks = info.get("communicate_checks") or []
    communicate_applicable = bool(checks) or "COMMUNICATE" in basis
    communicate_partial = None
    communicate_all = None
    if checks:
        passed = sum(bool(check.get("met")) for check in checks)
        communicate_partial = passed / len(checks)
        communicate_all = passed == len(checks)
    elif communicate_applicable:
        value = breakdown.get("COMMUNICATE", 1.0)
        communicate_partial = float(value)
        communicate_all = value == 1.0

    return OfficialScores(
        reward=float(official_reward),
        db_applicable=db_applicable,
        db_score=(float(db_value) if db_applicable and db_value is not None else None),
        communicate_applicable=communicate_applicable,
        communicate_partial=communicate_partial,
        communicate_all=communicate_all,
        raw_reward_info=info,
    )

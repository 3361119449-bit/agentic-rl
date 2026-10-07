"""Opt-in live update audit, with CPU-testable statistics and hash assertions.

This checks the whole update boundary. It deliberately does not claim access to
the engine worker's internal epoch boundaries.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from tau2_agentic_rl.versions import sha256_json


def ratio_statistics(current, old, mask):
    """Measure pi_actor / pi_rollout; backend differences need not be zero."""
    if not (len(current) == len(old) == len(mask)):
        raise ValueError("ratio inputs have different lengths")
    selected = [(a, b) for a, b, m in zip(current, old, mask, strict=True) if m]
    if not selected or any(
        not math.isfinite(a) or not math.isfinite(b) for a, b in selected
    ):
        raise ValueError("no policy tokens or non-finite policy log probabilities")
    if any(m not in (0, 1) for m in mask):
        raise ValueError("policy mask must be binary")
    try:
        ratios = sorted(math.exp(a - b) for a, b in selected)
    except OverflowError as exc:
        raise ValueError("non-finite PPO ratios") from exc
    if not all(map(math.isfinite, ratios)):
        raise ValueError("non-finite PPO ratios")
    mean = sum(ratios) / len(ratios)

    def percentile(q):
        index = q * (len(ratios) - 1)
        lo, hi = math.floor(index), math.ceil(index)
        return ratios[lo] + (ratios[hi] - ratios[lo]) * (index - lo)

    return {
        "policy_tokens": len(ratios),
        "log_prob_abs_diff_mean": sum(abs(a - b) for a, b in selected) / len(selected),
        "ratio_mean": mean,
        "ratio_std": math.sqrt(sum((r - mean) ** 2 for r in ratios) / len(ratios)),
        "ratio_p01": percentile(0.01),
        "ratio_p50": percentile(0.5),
        "ratio_p99": percentile(0.99),
        "ratio_max_abs_deviation": max(abs(r - 1) for r in ratios),
    }


def audit_update(
    *, read_old, compute_current, update, path: Path, tolerance: float = 0.005
):
    """Audit fixed behavior-policy probabilities without requiring backend parity.

    ``tolerance`` is diagnostic only, including for existing launch overrides.
    vLLM sampled the actions, so its log-probs remain the PPO denominator even
    when the FSDP actor assigns different probabilities before the update.
    """
    old_rows, mask_rows = read_old()
    old = [x for row in old_rows for x in row]
    mask = [x for row in mask_rows for x in row]
    current = [x for row in compute_current() for x in row]
    report = ratio_statistics(current, old, mask)
    report.update(
        {
            "behavior_policy_source": "vllm_rollout",
            "log_prob_abs_diff_tolerance": tolerance,
            "log_prob_abs_diff_exceeds_tolerance": (
                report["log_prob_abs_diff_mean"] > tolerance
            ),
            "old_log_probs_sha256_before_update": sha256_json(old_rows),
            "epoch_boundaries_instrumented": False,
            "status": "pre_update",
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if any(m == 0 and p != 0 for p, m in zip(old, mask, strict=True)):
        raise ValueError("non-policy old log probabilities must be zero")
    output = update()
    after_rows, after_mask = read_old()
    report["old_log_probs_sha256_after_update"] = sha256_json(after_rows)
    equal = after_rows == old_rows and after_mask == mask_rows
    report["status"] = "update_boundary_passed" if equal else "old_log_probs_changed"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not equal:
        raise ValueError(
            "old log probabilities or policy mask changed during PPO update"
        )
    return output, report

"""Queue adapters shared by the bounded sampler, PPO hook and offline replay."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from tau2_agentic_rl.advantages import CreditConfig, compute_group_credit
from tau2_agentic_rl.versions import sha256_json


def _save_group_audit(root: Path, report: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{sha256_json(report['uid'])}.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != report:
            raise ValueError("group audit identity or membership changed")
        return
    descriptor, name = tempfile.mkstemp(prefix=".group-", suffix=".tmp", dir=root)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def queue_group_reports(
    keys: list[str],
    extras: list,
    config: CreditConfig,
    *,
    audit_dir: Path | None = None,
    group_size: int = 8,
) -> dict[str, dict]:
    """Use the pinned queue key's uid/session, never the task ID as group ID."""
    if len(keys) != len(extras) or len(set(keys)) != len(keys):
        raise ValueError("queue group keys and extra fields do not align")
    groups = defaultdict(list)
    for key, extra in zip(keys, extras, strict=True):
        parts = key.split("_")
        if len(parts) != 3 or not parts[0] or not all(p.isdigit() for p in parts[1:]):
            raise ValueError("invalid pinned queue uid/session/index key")
        extra = getattr(extra, "data", extra)
        if not isinstance(extra, dict) or not isinstance(extra.get("procredit"), dict):
            raise ValueError("completed group is missing ProCredit scoring inputs")
        row = extra["procredit"]
        metric = extra.get("reward_extra_info", {}).get("train_reward", row["score"])
        if metric != row["score"]:
            raise ValueError("queue scalar score differs from credit score")
        groups[parts[0]].append((int(parts[1]), key, row))
    reports = {}
    for uid, members in groups.items():
        if len(members) != group_size:
            raise ValueError(f"incomplete ProCredit group {uid}: expected {group_size}")
        members.sort(key=lambda item: item[0])
        if [m[0] for m in members] != list(range(group_size)):
            raise ValueError("group session slots must occur exactly once")
        rows = [member[2] for member in members]
        credit = compute_group_credit(rows, config)
        report = {
            "version": "procredit-group-v1",
            "uid": uid,
            "members": [member[1] for member in members],
            "inputs": rows,
            "inputs_fingerprint": sha256_json(rows),
            "credit": credit,
            "decision": "has_signal" if credit["has_signal"] else "all_zero_advantage",
        }
        report["audit_fingerprint"] = sha256_json(report)
        if audit_dir is not None:
            _save_group_audit(Path(audit_dir), report)
        reports[uid] = report
    return reports


def credit_from_record(record) -> dict:
    """Rebuild only from a fully scored versioned record, never terminal guesses."""
    from tau2_agentic_rl.reward.policy_credit import build_policy_credit
    from tau2_agentic_rl.reward.progress import validate_progress_trace

    reward, trace = record.custom_reward, record.progress_trace
    if (
        record.schema_version != "2.0"
        or reward is None
        or reward.reward_mode != "strict_progress_v1"
        or trace is None
        or record.response_turn_ids is None
    ):
        raise ValueError("record lacks complete ProCredit scoring inputs")
    validate_progress_trace(trace, turns=record.assistant_turns)
    valid = reward.policy_gate and reward.task_safety_gate
    multiplier = reward.details["truncation_multiplier"]
    expected_score = (
        max(
            0,
            multiplier * (reward.strict_success + 0.5 * trace["phi"][-1])
            - reward.process_penalty,
        )
        if valid
        else 0
    )
    if (
        reward.details.get("checkset_fingerprint") != trace["checkset_fingerprint"]
        or reward.progress != trace["phi"][-1]
        or not math.isclose(
            reward.train_reward, expected_score, rel_tol=0, abs_tol=1e-12
        )
    ):
        raise ValueError("record reward differs from verified ProCredit inputs")
    turns = record.token_turns
    if not turns or len(turns) != record.assistant_turns:
        raise ValueError("record lacks original policy turns")
    initial_length = len(turns[0].prompt_token_ids)
    mapping = record.response_turn_ids
    if len(mapping) != record.trajectory_tokens - initial_length:
        raise ValueError("record token-to-turn mapping has a different response length")
    expected_map, previous_end = [-1] * len(mapping), 0
    for index, turn in enumerate(turns):
        start = len(turn.prompt_token_ids) - initial_length
        end = start + len(turn.output_token_ids)
        if (
            turn.assistant_turn_index != index
            or start < previous_end
            or end > len(mapping)
        ):
            raise ValueError("record policy turn boundaries changed")
        expected_map[start:end] = [index] * (end - start)
        previous_end = end
    if mapping != expected_map or not any(t >= 0 for t in mapping):
        raise ValueError(
            "record token-to-turn mapping differs from original policy tokens"
        )
    row = {
        "trajectory_id": record.trajectory_id,
        "task_id": record.task_id,
        "policy_version": record.policy_version,
        "checkset_fingerprint": trace["checkset_fingerprint"],
        "initial_state_fingerprint": trace["initial_state_fingerprint"],
        "score": reward.train_reward,
        "valid": valid,
        "terminal_success": reward.strict_success,
        "multiplier": multiplier,
        "phi": trace["phi"],
        "response_turn_ids": record.response_turn_ids,
    }
    version = reward.details.get("credit_version", "procredit-turn-v1")
    if version in {"procredit-turn-v2", "procredit-turn-v3"}:
        if record.judge_result is None:
            raise ValueError("policy-local credit requires a frozen Judge result")
        expected_policy = build_policy_credit(
            reward=reward, judge=record.judge_result, events=record.tool_events,
            turns=record.assistant_turns,
        )
        if expected_policy != reward.details.get("policy_credit"):
            raise ValueError("record policy attribution differs from frozen scoring evidence")
        row["policy_credit"] = expected_policy
    if version == "procredit-turn-v3":
        from dataclasses import asdict

        from tau2_agentic_rl.reward.process_penalty import build_process_credit
        from tau2_agentic_rl.reward.score import build_reward_config

        frozen = record.scoring_inputs.get("reward_project_config", {})
        config = build_reward_config(frozen)
        if (
            config.credit_version != version
            or trace["version"] != "progress-v2"
            or reward.details.get("process_config") != asdict(config.process)
        ):
            raise ValueError("record process configuration differs from frozen inputs")
        expected_process = build_process_credit(
            record.tool_events, record.assistant_turns, config.process,
        )
        if (
            expected_process != reward.details.get("process_credit")
            or not math.isclose(expected_process["total_cost"], reward.process_penalty, abs_tol=1e-12)
        ):
            raise ValueError("record process attribution differs from frozen scoring evidence")
        row["process_credit"] = expected_process
    return row


def build_credit_tensors(
    keys: list[str],
    extras: list,
    response_mask,
    config: CreditConfig,
    *,
    audit_dir: Path | None = None,
):
    """Create response-shaped float32 advantages without touching old log-probs."""
    import torch

    reports = queue_group_reports(keys, extras, config, audit_dir=audit_dir)
    tokens_by_key, mapping_by_key = {}, {}
    for report in reports.values():
        for key, row, result in zip(
            report["members"],
            report["inputs"],
            report["credit"]["trajectories"],
            strict=True,
        ):
            tokens_by_key[key] = result["token_advantages"]
            mapping_by_key[key] = row["response_turn_ids"]
    masks = list(response_mask.unbind())
    if len(masks) != len(keys):
        raise ValueError("response mask batch size differs from queue keys")
    tensors, active, nonzero = [], 0, 0
    for key, mask in zip(keys, masks, strict=True):
        mapping, values = mapping_by_key[key], tokens_by_key[key]
        mask_values = mask.detach().cpu().tolist()
        expected_mask = [int(turn >= 0) for turn in mapping]
        if response_mask.is_nested:
            aligned = mask_values == expected_mask
        else:
            aligned = (
                mask_values[: len(mapping)] == expected_mask
                and all(value == 0 for value in mask_values[len(mapping) :])
                and len(mask_values) >= len(mapping)
            )
        if not aligned:
            raise ValueError("response mask and token-to-turn mapping differ")
        if not response_mask.is_nested:
            values = values + [0.0] * (len(mask_values) - len(values))
        tensors.append(torch.tensor(values, dtype=torch.float32, device=mask.device))
        active += sum(expected_mask)
        nonzero += sum(abs(value) > config.signal_tolerance for value in values)
    advantages = (
        torch.nested.as_nested_tensor(tensors, layout=torch.jagged)
        if response_mask.is_nested
        else torch.stack(tensors)
    )
    metrics = {
        "procredit/nonzero_token_fraction": nonzero / active if active else 0.0,
        "procredit/constant_score_with_turn_signal": sum(
            report["credit"]["score_std"] == 0 and report["credit"]["has_signal"]
            for report in reports.values()
        ),
        "procredit/valid_turns": sum(
            report["credit"]["valid_turns"] for report in reports.values()
        ),
    }
    if config.version != "procredit-turn-v1":
        metrics.update({
            "procredit/policy_violation_turns": sum(
                report["credit"]["policy_violation_turns"] for report in reports.values()
            ),
            "procredit/unresolved_trajectories": sum(
                report["credit"]["unresolved_trajectories"] for report in reports.values()
            ),
        })
    if config.version == "procredit-turn-v3":
        metrics["procredit/process_cost_turns"] = sum(
            report["credit"]["process_cost_turns"] for report in reports.values()
        )
    return {"advantages": advantages, "returns": advantages.clone()}, metrics

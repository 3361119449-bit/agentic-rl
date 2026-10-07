"""Read frozen ProCredit v4 smoke data without generation or GPU dependencies."""

from __future__ import annotations

import json
import math
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from tau2_agentic_rl.advantages import CreditConfig, compute_group_credit
from tau2_agentic_rl.procredit_runtime import terminal_group_contract
from tau2_agentic_rl.versions import sha256_json

VERL_COMMIT = "483b8a009ba3a97563edee3a19887e4862b8094a"


@dataclass
class SavedTrajectory:
    trajectory_id: str
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    old_log_probs: list[float]
    advantages: list[float]
    trimmed_tail_tokens: int


@dataclass
class SavedBatch:
    trajectories: list[SavedTrajectory]
    launch: dict
    base_identity: dict
    manifest: dict


def _numbers(values, label):
    if not isinstance(values, list) or any(
        type(v) not in (int, float) or not math.isfinite(v) for v in values
    ):
        raise ValueError(f"invalid or non-finite {label}")


def _tokens(values):
    if not isinstance(values, list) or any(type(v) is not int or v < 0 for v in values):
        raise ValueError("invalid original token IDs")


def restore_trajectory(record, row, credit) -> SavedTrajectory:
    """Keep all generation prefixes; omit only an unsaved, zero-loss suffix."""
    for key in ("trajectory_id", "task_id", "policy_version", "response_turn_ids"):
        if record.get(key) != row.get(key):
            raise ValueError(f"trajectory differs from frozen credit input: {key}")
    if (
        record.get("schema_version") != "2.0"
        or record.get("split") != "train"
        or type(record.get("policy_version")) is not int
        or record["policy_version"] != 0
    ):
        raise ValueError(
            "replay supports only train schema 2.0 initial-policy version 0"
        )
    if record.get("user_sim_result", {}).get("user_sim_valid") is not True:
        raise ValueError("invalid user simulation cannot enter the PPO batch")
    if credit.get("trajectory_id") != record["trajectory_id"]:
        raise ValueError("credit trajectory identity mismatch")
    turns, mapping, advantages = (
        record["token_turns"],
        row["response_turn_ids"],
        credit["token_advantages"],
    )
    if (
        not turns
        or len(turns) != record["assistant_turns"]
        or not isinstance(mapping, list)
    ):
        raise ValueError("missing original generation turns or token mapping")
    _numbers(advantages, "advantages")
    initial = turns[0]["prompt_token_ids"]
    sequence = turns[-1]["prompt_token_ids"] + turns[-1]["output_token_ids"]
    _tokens(initial)
    _tokens(sequence)
    size = len(sequence) - len(initial)
    if (
        not initial
        or size <= 0
        or len(mapping) != len(advantages)
        or size > len(mapping)
        or record["trajectory_tokens"] != len(initial) + len(mapping)
    ):
        raise ValueError("original response lengths differ")
    expected, old, previous_end = [-1] * len(mapping), [0.0] * size, 0
    for index, turn in enumerate(turns):
        prompt, output, log_probs = (
            turn["prompt_token_ids"],
            turn["output_token_ids"],
            turn["output_old_log_probs"],
        )
        _tokens(prompt)
        _tokens(output)
        _numbers(log_probs, "rollout log probabilities")
        start, end = (
            len(prompt) - len(initial),
            len(prompt) - len(initial) + len(output),
        )
        if (
            turn["assistant_turn_index"] != index
            or not output
            or start < previous_end
            or end > size
            or len(output) != len(log_probs)
            or sequence[: len(prompt)] != prompt
            or sequence[len(prompt) : len(prompt) + len(output)] != output
        ):
            raise ValueError(
                "original policy prefix, output or log-prob alignment changed"
            )
        expected[start:end] = [index] * len(output)
        old[start:end] = log_probs
        previous_end = end
    if (
        mapping != expected
        or any(type(t) is not int for t in mapping)
        or not any(t >= 0 for t in mapping)
        or any(a != 0 for a, t in zip(advantages, mapping, strict=True) if t == -1)
    ):
        raise ValueError("policy mapping or zero-loss observation advantage changed")
    return SavedTrajectory(
        record["trajectory_id"],
        list(initial),
        sequence[len(initial) :],
        [int(t >= 0) for t in mapping[:size]],
        old,
        list(advantages[:size]),
        len(mapping) - size,
    )


def validate_group(report, *, group_size):
    """Validate the saved membership and recompute credit solely as an integrity check."""
    body = {k: v for k, v in report.items() if k != "audit_fingerprint"}
    if report.get("audit_fingerprint") != sha256_json(body):
        raise ValueError("group audit fingerprint changed")
    if report.get("version") != "procredit-group-v2":
        raise ValueError("replay requires settled v4 group membership")
    inputs, credit = report["inputs"], report["credit"]
    if report["inputs_fingerprint"] != sha256_json(inputs):
        raise ValueError("group inputs fingerprint changed")
    config = CreditConfig(**credit["config"])
    if (
        config.version != "procredit-turn-v4"
        or compute_group_credit(inputs, config) != credit
    ):
        raise ValueError("frozen ProCredit advantages differ from verified inputs")
    if report["decision"] != (
        "has_signal" if credit["has_signal"] else "all_zero_advantage"
    ):
        raise ValueError("group filtering decision changed")
    terminal = terminal_group_contract(report["terminal_group"], group_size)
    expected = [
        f"{report['uid']}_{i}_0"
        for i in range(group_size)
        if i not in terminal["failed_session_ids"]
    ]
    if report["members"] != expected or len(inputs) != len(expected):
        raise ValueError("group members differ from settled session slots")
    by_id = {c["trajectory_id"]: c for c in credit["trajectories"]}
    if len(by_id) != len(inputs) or {r["trajectory_id"] for r in inputs} != set(by_id):
        raise ValueError("duplicate or missing credit trajectory")
    return [(row, by_id[row["trajectory_id"]]) for row in inputs]


def _relevant(path):
    return (
        path
        in {
            "base_model_identity.json",
            "rl_resume_identity.json",
            "ppo_audit/step_1.json",
        }
        or re.fullmatch(r"(?:launches|group_audits|trajectories)/[^/]+\.json", path)
        is not None
    )


def read_bundle(source: Path) -> dict:
    """Read JSON directly from the tar: no extraction, symlinks or path traversal."""
    source = Path(source)
    if source.is_dir():
        if not (source / "base_model_identity.json").is_file():
            candidates = list(source.glob("*/base_model_identity.json"))
            if len(candidates) != 1:
                raise ValueError("results directory must identify exactly one run")
            source = candidates[0].parent
        return {
            p.relative_to(source).as_posix(): json.loads(p.read_text(encoding="utf-8"))
            for p in source.rglob("*.json")
            if _relevant(p.relative_to(source).as_posix())
        }
    with tarfile.open(source, "r:*") as archive:
        members = archive.getmembers()
        for member in members:
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or "\\" in member.name
                or member.issym()
                or member.islnk()
            ):
                raise ValueError("unsafe path or link in results archive")
        roots = [
            PurePosixPath(m.name).parent
            for m in members
            if m.isfile() and PurePosixPath(m.name).name == "base_model_identity.json"
        ]
        if len(roots) != 1:
            raise ValueError("results archive must identify exactly one run")
        result = {}
        for member in members:
            if not member.isfile():
                continue
            try:
                relative = PurePosixPath(member.name).relative_to(roots[0]).as_posix()
            except ValueError:
                continue
            if not _relevant(relative):
                continue
            if relative in result or member.size > 32 * 1024 * 1024:
                raise ValueError("duplicate or oversized saved result")
            with archive.extractfile(member) as handle:
                result[relative] = json.load(handle)
        return result


def load_saved_batch(source: Path) -> SavedBatch:
    from tau2_agentic_rl.procredit_runtime import credit_from_record
    from tau2_agentic_rl.schemas import TrajectoryRecord

    bundle = read_bundle(source)
    launches = [v for k, v in bundle.items() if k.startswith("launches/")]
    if len(launches) != 1:
        raise ValueError("replay requires exactly one saved launch")
    launch, identity = launches[0], bundle["base_model_identity.json"]
    project, command = launch["config"], launch["command"]
    if project["project"]["verl_commit"] != VERL_COMMIT or command[1:3] != [
        "-m",
        "tau2_agentic_rl.verl_entrypoint",
    ]:
        raise ValueError("saved launch is not the supported pinned veRL training flow")
    if (
        identity.get("schema_version") != 1
        or not identity.get("files")
        or not identity.get("chat_template_sha256")
    ):
        raise ValueError("missing complete original base-model identity")
    audit = bundle["ppo_audit/step_1.json"]
    if audit.get("status") != "pre_update":
        raise ValueError(
            "results must describe the initial batch stopped before update"
        )
    group_size = project["rollout"]["group_size"]
    trajectories, groups, seen = [], [], set()
    reports = sorted(
        (v for k, v in bundle.items() if k.startswith("group_audits/")),
        key=lambda v: v["uid"],
    )
    for report in reports:
        pairs = validate_group(report, group_size=group_size)
        if (
            report["credit"]["config"]
            != CreditConfig.from_project(project).audit_config()
        ):
            raise ValueError("group credit configuration differs from saved launch")
        if not report["credit"]["has_signal"]:
            continue
        if report["uid"] in {g["uid"] for g in groups}:
            raise ValueError("duplicate group uid")
        groups.append(
            {
                "uid": report["uid"],
                "audit_fingerprint": report["audit_fingerprint"],
                "members": report["members"],
            }
        )
        for row, credit in pairs:
            trajectory_id = row["trajectory_id"]
            if (
                not re.fullmatch(r"[A-Za-z0-9_-]+", trajectory_id)
                or trajectory_id in seen
            ):
                raise ValueError("duplicate or invalid trajectory identity")
            record = bundle[f"trajectories/{trajectory_id}.json"]
            if credit_from_record(TrajectoryRecord.model_validate(record)) != row:
                raise ValueError(
                    "trajectory scoring evidence differs from saved credit inputs"
                )
            trajectories.append(restore_trajectory(record, row, credit))
            seen.add(trajectory_id)
    count = sum(sum(t.response_mask) for t in trajectories)
    if not trajectories or count != audit["policy_tokens"]:
        raise ValueError("saved groups do not reproduce the audited initial PPO batch")
    mini_batch = project["ppo"]["verl_ppo_mini_batch_size_prompts"] * group_size
    if type(mini_batch) is not int or mini_batch <= 0:
        raise ValueError("invalid saved PPO mini-batch size")
    padding = (-len(trajectories)) % mini_batch
    manifest = {
        "schema_version": 1,
        "source": str(Path(source).resolve()),
        "policy_version": 0,
        "groups": groups,
        "real_trajectories": len(trajectories),
        "policy_tokens": count,
        "retained_response_tokens": sum(len(t.response_ids) for t in trajectories),
        "trimmed_nonpolicy_tail_tokens": sum(
            t.trimmed_tail_tokens for t in trajectories
        ),
        "padding_rows": padding,
        "ppo_mini_batch_size": mini_batch,
        "ppo_epochs": project["ppo"]["ppo_epochs"],
        "outer_update_steps": 1,
        "expected_optimizer_iterations": (len(trajectories) + padding)
        // mini_batch
        * project["ppo"]["ppo_epochs"],
        "batch_sha256": sha256_json([t.__dict__ for t in trajectories]),
        "launch_sha256": sha256_json(launch),
        "base_identity_sha256": sha256_json(identity),
        "source_ppo_audit": audit,
        "base_model_verified": False,
        "row_order": "group uid, then original session slot; source queue order was not saved",
    }
    return SavedBatch(trajectories, launch, identity, manifest)


def verify_base_model(batch: SavedBatch, model_path: Path):
    from tau2_agentic_rl.base_identity import capture_base_identity

    if any(
        (Path(model_path) / name).exists()
        for name in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
        )
    ):
        raise ValueError("original merged SFT base must not contain adapter artifacts")
    actual = capture_base_identity(model_path)
    for key in ("schema_version", "files", "chat_template_sha256"):
        if actual[key] != batch.base_identity[key]:
            raise ValueError(
                f"model differs from the policy that generated the saved batch: {key}"
            )
    batch.manifest["base_model_verified"] = True
    return actual

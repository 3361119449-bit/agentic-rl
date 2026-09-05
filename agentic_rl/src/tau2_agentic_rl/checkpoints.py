"""Validate explicit RL checkpoints and restore their own audit counters."""

from __future__ import annotations

import json
import re
from pathlib import Path

from tau2_agentic_rl.dynamic_sampling import TrainingStepClock


def resolve_resume_path(project_root: Path, candidate: Path) -> Path:
    path = (
        candidate if candidate.is_absolute() else project_root / candidate
    ).resolve()
    if not path.is_dir() or not re.fullmatch(r"global_step_\d+", path.name):
        raise ValueError(f"not an RL global_step_N checkpoint: {path}")
    required = [path / "data.pt", path / "actor"]
    if any(not item.exists() for item in required):
        raise FileNotFoundError(f"checkpoint requires actor/ and data.pt: {path}")
    for prefix in ("model", "optim", "extra_state"):
        if not any((path / "actor").glob(f"{prefix}*_rank_*.pt")):
            raise FileNotFoundError(f"missing actor {prefix} FSDP shards: {path}")
    return path


def restore_step_clock(checkpoint: Path, optimizer_step: int) -> TrainingStepClock:
    path = checkpoint / "step_counters.json"
    if not path.is_file():
        # Compatibility with c4883f7: only accept the legacy run-level file if
        # it describes exactly this checkpoint, never a later update.
        path = checkpoint.parent.parent / "step_counters.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing step counters for checkpoint: {checkpoint}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    clock = TrainingStepClock(**payload)
    if (
        clock.optimizer_step != optimizer_step
        or clock.attempt_step < clock.optimizer_step
        or clock.consecutive_skips < 0
        or clock.consecutive_skips > clock.attempt_step - clock.optimizer_step
    ):
        raise ValueError("step counters do not match the selected checkpoint")
    return clock

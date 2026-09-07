"""Bind RL continuation to the launch settings and input contents, not paths."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from tau2_agentic_rl.versions import sha256_file

MANIFEST = "rl_resume_identity.json"

# Only relocation/continuation controls are omitted. Seeds, epochs, scheduler
# settings and even unmapped --extra training overrides remain identity-bound.
RELOCATABLE_OVERRIDES = {
    "trainer.resume_mode",
    "trainer.resume_from_path",
    "trainer.default_local_dir",
    "trainer.experiment_name",
    "trainer.rollout_data_dir",
    "actor_rollout_ref.model.path",  # Independently checked by base identity.
}
INPUT_OVERRIDES = {
    "data.train_files",
    "data.val_files",
    "actor_rollout_ref.rollout.agent.agent_loop_config_path",
}


def build_resume_identity(
    project_root: Path, project: dict, command: list[str], *, stage: str
) -> dict:
    """Capture the final launch command, including the last effective override."""
    overrides = {}
    for argument in command[3:]:
        key, separator, value = argument.partition("=")
        if not separator:
            raise ValueError(f"cannot record RL override: {argument}")
        overrides[key.lstrip("+")] = value

    def fingerprint(value):
        path = Path(value)
        if not path.is_absolute():
            path = project_root / path
        return sha256_file(path)

    files = {key: fingerprint(overrides.pop(key)) for key in INPUT_OVERRIDES}
    for key in RELOCATABLE_OVERRIDES:
        overrides.pop(key, None)
    runtime = deepcopy(project)
    runtime.pop("outputs", None)
    runtime.get("model", {}).pop("sft_input", None)
    for name, value in runtime.get("annotations", {}).items():
        digest = fingerprint(value)
        files[f"annotations.{name}"] = digest
        runtime["annotations"][name] = {"sha256": digest}
    return {
        "schema_version": 1,
        "stage": stage,
        "runtime_config": runtime,
        "training_overrides": overrides,
        "files": files,
    }


def read_resume_identity(directory: Path) -> dict:
    path = directory / MANIFEST
    if not path.is_file():
        raise FileNotFoundError(
            f"missing RL resume identity: {path}; legacy or incomplete checkpoints "
            "cannot resume optimizer/dataloader state. Export/merge weights and "
            "start a new run without --resume-from-path instead."
        )
    identity = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(identity, dict)
        or identity.get("schema_version") != 1
        or not {"stage", "runtime_config", "training_overrides", "files"}
        <= identity.keys()
    ):
        raise ValueError(f"invalid RL resume identity: {path}")
    return identity


def require_same_identity(directory: Path, expected: dict) -> None:
    saved = read_resume_identity(directory)
    if saved != expected:
        changed = sorted(
            key
            for key in saved.keys() | expected.keys()
            if saved.get(key) != expected.get(key)
        )
        raise ValueError(
            "RL resume identity changed: "
            + ", ".join(changed)
            + "; restore the original stage, data, seed, epochs and overrides. "
            "A new output directory does not start a new optimizer experiment."
        )


def save_resume_identity(directory: Path, identity: dict) -> None:
    path = directory / MANIFEST
    if path.exists():
        require_same_identity(directory, identity)
        return
    directory.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(identity, handle, ensure_ascii=False, indent=2, sort_keys=True)


def snapshot_resume_identity(run_root: Path, checkpoint: Path) -> None:
    """Each completed checkpoint carries its own immutable launch identity."""
    save_resume_identity(checkpoint, read_resume_identity(run_root))

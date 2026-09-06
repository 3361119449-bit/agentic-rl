"""Content/configuration-bound, epoch-boundary LoRA SFT checkpoints.

This module is stdlib-only so the launcher can reject bad resumes before CUDA.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

VERL_COMMIT = "bec9ef74768dd201881cd4e54cd0385e87caae27"
RUN_MANIFEST = "sft_run_identity.json"
COMPLETE_MANIFEST = "sft_checkpoint_complete.json"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_run_identity(directory, identity):
    path = Path(directory) / RUN_MANIFEST
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != identity:
            raise ValueError(
                "SFT run identity changed; use a fresh run for a new experiment"
            )
    else:
        write_json_atomic(path, identity)


def validate_runtime():
    """Validate the actual imported checkout, not a different --verl-root path."""
    spec = importlib.util.find_spec("verl")
    if spec is None or spec.origin is None:
        raise RuntimeError("Install the pinned SFT veRL checkout; see the SFT README")
    root = Path(spec.origin).resolve().parent.parent
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if commit != VERL_COMMIT or dirty:
        raise RuntimeError(
            f"SFT requires clean veRL {VERL_COMMIT}; found {commit}, dirty={bool(dirty)}"
        )
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "verl",
            "transformers",
            "torch",
            "torchdata",
            "peft",
            "tensordict",
            "pyarrow",
        )
    }
    if versions["transformers"] != "4.57.1":
        raise RuntimeError(
            "This SFT tokenizer/LoRA contract requires transformers==4.57.1"
        )
    return {"verl_commit": commit, "packages": versions}


def build_run_identity(args, prepared, train_path, val_path, base_identity, runtime):
    fields = (
        "lora_rank",
        "lora_alpha",
        "num_gpus",
        "ulysses_size",
        "fsdp_strategy",
        "global_batch_size",
        "micro_batch_size",
        "max_length",
        "max_token_len_per_gpu",
        "epochs",
        "warmup_ratio",
        "weight_decay",
        "param_offload",
        "optimizer_offload",
        "num_workers",
        "val_ratio",
        "seed",
        "extra_config",
    )
    config = {key: getattr(args, key) for key in fields}
    config["learning_rate"] = (
        args.learning_rate if args.learning_rate is not None else 1e-4
    )
    training_dir = Path(__file__).resolve().parents[3] / "training/qwen3_4b_sft"
    return {
        "schema_version": 1,
        "config": config,
        "base_files": base_identity["files"],
        "data": {
            "source_sha256": prepared["input_sha256"],
            "train_sha256": file_sha256(train_path),
            "validation_sha256": file_sha256(val_path) if val_path else None,
            "train_rows": prepared["train_rows"],
            "validation_rows": prepared["validation_rows"],
            "normalization_version": prepared["script_version"],
        },
        "runtime": runtime,
        "implementation": {
            name: file_sha256(training_dir / name)
            for name in ("train_qwen3_4b_verl_sft.py", "lora_sft_runtime.py")
        }
        | {"sft_contract.py": file_sha256(__file__)},
        "loss_scope": "last_assistant_answer_only",
        "validation": "replicated_per_example_mean_answer_nll",
        "checkpoint_boundary": "completed_epoch",
    }


def checkpoint_files(identity):
    config = identity["config"]
    world = config["num_gpus"]
    dp = world // config["ulysses_size"]
    names = [
        f"{prefix}_world_size_{world}_rank_{rank}.pt"
        for rank in range(world)
        for prefix in ("model", "optim", "extra_state")
    ]
    return (
        names
        + [f"data_{rank}.pt" for rank in range(dp)]
        + [
            "lora_train_meta.json",
            "fsdp_config.json",
            "huggingface/config.json",
            "huggingface/tokenizer_config.json",
            "huggingface/tokenizer.json",
            RUN_MANIFEST,
            "base_model_identity.json",
        ]
    )


def checkpoint_inventory(path, identity):
    inventory = {}
    for name in checkpoint_files(identity):
        item = Path(path) / name
        if not item.is_file() or item.stat().st_size == 0:
            raise FileNotFoundError(f"Incomplete SFT checkpoint: {item}")
        # Hash small loader/metadata files. Large model/optimizer shards are
        # checked for presence and size; this is not a full corruption audit.
        inventory[name] = {"size": item.stat().st_size}
        if not re.match(r"(?:model|optim|extra_state)_world_size_", name):
            inventory[name]["sha256"] = file_sha256(item)
    return inventory


def publish_checkpoint(path, identity, step):
    path = Path(path)
    if (path / COMPLETE_MANIFEST).exists():
        raise FileExistsError(f"Refusing to replace a completed checkpoint: {path}")
    save_run_identity(path, identity)
    write_json_atomic(
        path / COMPLETE_MANIFEST,
        {
            "schema_version": 1,
            "step": step,
            "run_identity_sha256": identity_hash(identity),
            "files": checkpoint_inventory(path, identity),
        },
    )


def validate_resume(path, identity):
    path = Path(path).resolve()
    match = re.fullmatch(r"global_step_(\d+)", path.name)
    if not path.is_dir() or match is None:
        raise ValueError(f"Expected an SFT global_step_N directory: {path}")
    marker = path / COMPLETE_MANIFEST
    if not marker.is_file() or not (path / RUN_MANIFEST).is_file():
        raise FileNotFoundError(
            "Missing SFT completion/run manifest; legacy or partial checkpoints cannot resume"
        )
    saved = json.loads((path / RUN_MANIFEST).read_text(encoding="utf-8"))
    if saved != identity:
        raise ValueError(
            "SFT resume identity mismatch (LoRA/config/data/base/runtime/code); restore the original settings"
        )
    step = int(match[1])
    steps_per_epoch = (
        identity["data"]["train_rows"] // identity["config"]["global_batch_size"]
    )
    if steps_per_epoch < 1 or step <= 0 or step % steps_per_epoch:
        raise ValueError("Only completed-epoch SFT checkpoints can resume")
    if step >= steps_per_epoch * identity["config"]["epochs"]:
        raise ValueError(
            "This SFT run is already complete; start a fresh stage instead of resuming"
        )
    complete = json.loads(marker.read_text(encoding="utf-8"))
    if (
        complete.get("schema_version") != 1
        or complete.get("step") != step
        or complete.get("run_identity_sha256") != identity_hash(identity)
        or complete.get("files") != checkpoint_inventory(path, identity)
    ):
        raise ValueError("SFT checkpoint completion/inventory mismatch")
    meta = json.loads((path / "lora_train_meta.json").read_text(encoding="utf-8"))
    if (
        meta.get("r") != identity["config"]["lora_rank"]
        or meta.get("lora_alpha") != identity["config"]["lora_alpha"]
        or meta.get("task_type") != "CAUSAL_LM"
    ):
        raise ValueError("Checkpoint LoRA metadata mismatch")
    return path, step


def validate_output(directory, identity, resume_path=None):
    directory = Path(directory).resolve()
    if not directory.exists() or not any(directory.iterdir()):
        return
    if resume_path is None:
        raise FileExistsError(f"Output directory is not empty: {directory}")
    saved_path = directory / RUN_MANIFEST
    if (
        not saved_path.is_file()
        or json.loads(saved_path.read_text(encoding="utf-8")) != identity
    ):
        raise ValueError("Output directory belongs to a different/untracked SFT run")
    _, step = validate_resume(resume_path, identity)
    for child in directory.iterdir():
        match = re.fullmatch(r"global_step_(\d+)", child.name)
        if match and int(match[1]) > step:
            raise FileExistsError(
                "Later checkpoint directories exist; resume into a NEW output directory"
            )

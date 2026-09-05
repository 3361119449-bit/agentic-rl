"""Export a standard PEFT adapter from a pinned veRL FSDP checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from tau2_agentic_rl.base_identity import find_base_identity, save_base_identity

VERL_COMMITS = {
    "sft": "bec9ef74768dd201881cd4e54cd0385e87caae27",  # v0.7.1
    "rl": "483b8a009ba3a97563edee3a19887e4862b8094a",  # v0.9.0
}


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def adapter_dir(target_dir: Path) -> Path:
    """Return and validate model_merger's standard LoRA output directory."""
    candidate = target_dir / "lora_adapter"
    required = [
        candidate / "adapter_config.json",
        candidate / "adapter_model.safetensors",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "verl.model_merger did not produce a complete PEFT adapter: "
            + ", ".join(missing)
        )
    return candidate


def build_command(local_dir: Path, target_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(local_dir.resolve()),
        "--target_dir",
        str(target_dir.resolve()),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(VERL_COMMITS), required=True)
    parser.add_argument("--verl-root", type=Path, required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--base-identity",
        type=Path,
        help="Original training manifest if checkpoint was moved",
    )
    args = parser.parse_args()

    actual = _git_commit(args.verl_root)
    expected = VERL_COMMITS[args.stage]
    if actual != expected:
        raise RuntimeError(
            f"veRL commit mismatch for {args.stage}: expected {expected}, got {actual}"
        )
    if not args.local_dir.is_dir():
        raise FileNotFoundError(args.local_dir)
    if not args.dry_run and args.target_dir.exists() and any(args.target_dir.iterdir()):
        raise FileExistsError(f"target directory is not empty: {args.target_dir}")

    command = build_command(args.local_dir, args.target_dir)
    print(shlex.join(command))
    if args.dry_run:
        return
    identity_path = args.base_identity or find_base_identity(args.local_dir)
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("schema_version") != 1 or not identity.get("files"):
        raise ValueError("invalid training base identity")
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(args.verl_root), *([existing] if existing else [])]
    )
    subprocess.run(command, check=True, cwd=args.verl_root, env=environment)
    adapter = adapter_dir(args.target_dir)
    save_base_identity(adapter, identity)
    print(adapter)


if __name__ == "__main__":
    main()

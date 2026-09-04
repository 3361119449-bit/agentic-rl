"""Capture immutable source and package versions for one experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

from tau2_agentic_rl.config import load_runtime_config
from tau2_agentic_rl.judge.prompts import JUDGE_SYSTEM_PROMPT

PACKAGES = ("torch", "transformers", "vllm", "verl", "tau2-bench", "peft")


def _commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _tree_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _model_metadata(model_path: str | None) -> dict[str, object] | None:
    if not model_path:
        return None
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    chat_template = tokenizer.chat_template or ""
    local_path = Path(model_path)
    metadata_files = {}
    if local_path.is_dir():
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            path = local_path / name
            if path.exists():
                metadata_files[name] = _sha256_file(path)
    return {
        "path": model_path,
        "model_revision": getattr(config, "_commit_hash", None),
        "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "chat_template_sha256": _sha256_bytes(chat_template.encode("utf-8")),
        "metadata_files_sha256": metadata_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--verl-root", type=Path, required=True)
    parser.add_argument(
        "--project-config",
        type=Path,
        default=Path("configs/rl/airline_grpo_v1.yaml"),
    )
    parser.add_argument("--model-path", default=os.environ.get("MERGED_SFT_MODEL"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.project_config.resolve()
    config = load_runtime_config(config_path)
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tau2_commit": _commit(args.tau2_root),
        "verl_commit": _commit(args.verl_root),
        "packages": versions,
        "cuda": None,
        "model": _model_metadata(args.model_path),
        "user_simulator_model": config["user_simulator"]["model"],
        "judge_model": config["judge"]["model"],
        "judge_system_prompt_sha256": _sha256_bytes(
            JUDGE_SYSTEM_PROMPT.encode("utf-8")
        ),
        "project_config_sha256": _sha256_file(config_path),
        "annotation_files_sha256": {
            key: _sha256_file(project_root / relative)
            for key, relative in config["annotations"].items()
        },
        "reward_code_sha256": _tree_hash(
            list((project_root / "src/tau2_agentic_rl/reward").glob("*.py"))
        ),
        "per_trajectory_user_and_judge_prompt_hashes": True,
    }
    try:
        import torch

        payload["cuda"] = torch.version.cuda
    except ImportError:
        pass
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

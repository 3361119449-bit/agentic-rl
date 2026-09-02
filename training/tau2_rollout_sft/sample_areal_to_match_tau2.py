#!/usr/bin/env python3
"""Randomly sample cleaned AReaL rows to match a Tau2 SFT row count."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any


DEFAULT_AREAL = Path(
    "datasets/tau2_airline_sft_strict_cleaned/data/"
    "airline_sft_no_thinking_under_16k_strict_leakage_cleaned.jsonl"
)
FORBIDDEN_KEYS = {"thinking", "reasoning", "reasoning_content"}
THINK_TAG_RE = re.compile(r"</?think(?:\s[^>]*)?>", re.IGNORECASE)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--areal", type=Path, default=root / DEFAULT_AREAL)
    parser.add_argument("--tau2-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def nonempty_jsonl_lines(path: Path) -> list[bytes]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        return [line for line in handle if line.strip()]


def contains_reasoning(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                return True
            if contains_reasoning(child):
                return True
    elif isinstance(value, list):
        return any(contains_reasoning(child) for child in value)
    elif isinstance(value, str):
        return bool(THINK_TAG_RE.search(value))
    return False


def validate_areal_rows(lines: list[bytes]) -> None:
    for line_number, raw in enumerate(lines, 1):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid AReaL JSON on line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"AReaL line {line_number} is not a JSON object")
        if not isinstance(row.get("messages"), list):
            raise ValueError(f"AReaL line {line_number} has no messages list")
        answer = row.get("answer")
        if not isinstance(answer, dict) or answer.get("role") != "assistant":
            raise ValueError(f"AReaL line {line_number} has no assistant answer")
        if contains_reasoning(row):
            raise ValueError(f"AReaL line {line_number} contains reasoning/thinking")


def sha256_lines(lines: list[bytes]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line)
    return digest.hexdigest()


def write_atomic(path: Path, lines: list[bytes]) -> str:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for line in lines:
            normalized = line.rstrip(b"\r\n") + b"\n"
            handle.write(normalized)
            digest.update(normalized)
    os.replace(temporary, path)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    areal_path = args.areal.resolve()
    reference_path = args.tau2_reference.resolve()
    source_lines = nonempty_jsonl_lines(areal_path)
    reference_lines = nonempty_jsonl_lines(reference_path)
    validate_areal_rows(source_lines)

    sample_size = len(reference_lines)
    if sample_size == 0:
        raise ValueError("Tau2 reference JSONL is empty")
    if sample_size > len(source_lines):
        raise ValueError(
            f"Tau2 has {sample_size} rows but AReaL has only {len(source_lines)}; "
            "sampling without replacement is impossible"
        )

    indices = sorted(random.Random(args.seed).sample(range(len(source_lines)), sample_size))
    sampled_lines = [source_lines[index] for index in indices]
    output_sha256 = write_atomic(args.output, sampled_lines)
    manifest_path = (
        args.manifest.resolve()
        if args.manifest
        else args.output.resolve().with_suffix(".manifest.json")
    )
    manifest = {
        "areal_source": str(areal_path),
        "areal_source_rows": len(source_lines),
        "areal_source_sha256": sha256_lines(source_lines),
        "tau2_reference": str(reference_path),
        "tau2_reference_rows": sample_size,
        "tau2_reference_sha256": sha256_lines(reference_lines),
        "output": str(args.output.resolve()),
        "output_rows": len(sampled_lines),
        "output_sha256": output_sha256,
        "seed": args.seed,
        "sampling": "uniform_rows_without_replacement",
        "step_matching_requirement": (
            "Use the same val_ratio=0, epochs, global batch size, and optimizer "
            "settings as the AReaL-then-Tau2 second stage."
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

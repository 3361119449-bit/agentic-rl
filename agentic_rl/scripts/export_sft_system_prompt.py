"""Extract one byte-exact system-message string after checking EVERY SFT row.

No models or APIs. The input JSONL is read-only; output must be a new file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tau2_agentic_rl.agent_policy import extract_airline_policy, prompt_sha256

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / (
    "datasets/tau2_airline_sft_strict_cleaned/data/"
    "airline_sft_no_thinking_under_16k_strict_leakage_cleaned.jsonl"
)


def extract(source: Path) -> dict:
    prompt, count = None, 0
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            messages = row.get("messages", [])
            systems = [m for m in messages if m.get("role") == "system"]
            if len(systems) != 1 or not messages or messages[0] != systems[0]:
                raise ValueError(
                    f"row {line_number}: require exactly one leading system"
                )
            text = systems[0].get("content")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"row {line_number}: missing system text")
            if prompt is not None and text != prompt:
                raise ValueError(f"row {line_number}: multiple SFT system prompts")
            prompt = text  # Do not strip, reformat, or substitute policy text.
            count += 1
    if prompt is None:
        raise ValueError("empty SFT input")
    extract_airline_policy(prompt)
    path = source.resolve()
    return {
        "schema_version": 1,
        "source": str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        if path.is_relative_to(REPO_ROOT)
        else path.name,
        "source_sha256": digest.hexdigest(),
        "source_rows": count,
        "system_prompt_sha256": prompt_sha256(prompt),
        "system_prompt": prompt,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(
            "choose a new output; existing artifacts are not overwritten"
        )
    artifact = extract(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in artifact.items() if k != "system_prompt"}))


if __name__ == "__main__":
    main()

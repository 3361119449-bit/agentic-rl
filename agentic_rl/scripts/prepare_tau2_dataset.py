"""Prepare split-safe Tau2 task IDs in veRL's parquet row format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SMOKE_IDS = ["0", "4", "11", "14", "20", "28", "40", "46"]


def _read(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _row(task_id: str, split: str, index: int) -> dict[str, Any]:
    return {
        "data_source": "tau2_airline",
        "agent_name": "tau2_airline",
        # The custom agent loop builds the real policy/tool/user prompt from a
        # fresh Tau2 environment. This minimal prompt only satisfies RLHFDataset.
        "prompt": [
            {
                "role": "user",
                "content": f"Initialize isolated Tau2 Airline task {task_id}.",
            }
        ],
        "ability": "agentic_airline",
        "reward_model": {"style": "rule", "ground_truth": None},
        "extra_info": {
            "task_id": task_id,
            "split": split,
            "environment_seed": index,
        },
    }


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("install the project with the 'data' extra") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(path))


def prepare(split_file: Path, output_dir: Path) -> dict[str, int]:
    """Write RL-train, internal-dev, full-train, and frozen-test parquets."""
    split = _read(split_file)
    if set(split["rl_train"]) & set(split["internal_dev"]):
        raise ValueError("RL train and internal dev overlap")
    if set(SMOKE_IDS) - set(split["rl_train"]):
        raise ValueError("smoke tasks must be a strict subset of RL train")
    official_train = sorted(
        set(split["rl_train"]) | set(split["internal_dev"]), key=int
    )
    official_test = list(split["official_test"])
    if set(official_train) & set(official_test):
        raise ValueError("official train and test overlap")

    bundles = {
        "smoke": ([item for item in SMOKE_IDS if item in official_train], "train"),
        "rl_train": (split["rl_train"], "train"),
        "internal_dev": (split["internal_dev"], "internal_dev"),
        "official_train": (official_train, "train"),
        "official_test": (official_test, "test"),
    }
    counts = {}
    for name, (ids, label) in bundles.items():
        rows = [_row(str(task_id), label, index) for index, task_id in enumerate(ids)]
        _write_parquet(rows, output_dir / f"airline_{name}.parquet")
        counts[name] = len(rows)
    (output_dir / "manifest.json").write_text(
        json.dumps({"schema_version": "1.0", "counts": counts}, indent=2) + "\n",
        encoding="utf-8",
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("data/splits/airline_internal_dev.v1.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/parquet"))
    args = parser.parse_args()
    counts = prepare(args.split_file, args.output_dir)
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()

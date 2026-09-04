"""CPU-only acceptance checks for split isolation and annotation fidelity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

TAU2_COMMIT = "a2c024725189473d2d7cea3a5cfdbcc67478e41f"
VERL_COMMIT = "483b8a009ba3a97563edee3a19887e4862b8094a"


def _read(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def verify(project_root: Path, tau2_data: Path) -> dict[str, Any]:
    split = _read(tau2_data / "split_tasks.json")
    tasks = _read(tau2_data / "tasks.json")
    split_local = _read(project_root / "data/splits/airline_internal_dev.v1.json")
    train_ids = set(map(str, split["train"]))
    test_ids = set(map(str, split["test"]))
    assert not train_ids & test_ids
    assert set(split_local["rl_train"]) | set(split_local["internal_dev"]) == train_ids
    assert set(split_local["official_test"]) == test_ids
    assert not set(split_local["rl_train"]) & set(split_local["internal_dev"])

    official_actions = {
        str(task["id"]): {
            _canonical(action)
            for action in (task.get("evaluation_criteria") or {}).get("actions", [])
        }
        for task in tasks
    }
    selected_count = 0
    for name, expected_ids in (("train", train_ids), ("test", test_ids)):
        rows = _read(
            project_root / f"data/annotations/airline_required_actions.{name}.v1.json"
        )
        assert {str(row["id"]) for row in rows} == expected_ids
        for row in rows:
            for action in row["actions"]:
                assert _canonical(action) in official_actions[str(row["id"])]
                selected_count += 1

    config = yaml.safe_load(
        (project_root / "configs/rl/airline_grpo_v1.yaml").read_text(encoding="utf-8")
    )
    assert config["project"]["tau2_commit"] == TAU2_COMMIT
    assert config["project"]["verl_commit"] == VERL_COMMIT
    dynamic = config["dynamic_sampling"]
    assert (
        dynamic["conceptual_gen_batch_size"]
        * dynamic["group_size"]
        * dynamic["max_num_gen_batches"]
        == dynamic["max_rollouts_per_optimizer_step"]
    )
    assert sum(config["reward"]["normal_weights"].values()) == 1.0
    assert sum(config["reward"]["transfer_weights"].values()) == 1.0
    return {
        "official_train_tasks": len(train_ids),
        "rl_train_tasks": len(split_local["rl_train"]),
        "internal_dev_tasks": len(split_local["internal_dev"]),
        "official_test_tasks": len(test_ids),
        "selected_required_actions": selected_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--tau2-data", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.project_root, args.tau2_data), indent=2))


if __name__ == "__main__":
    main()

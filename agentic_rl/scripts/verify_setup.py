"""CPU-only acceptance checks for split isolation and annotation fidelity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from tau2_agentic_rl.policy_rules import policy_checks

try:
    from scripts.prepare_tau2_dataset import SMOKE_IDS
except ModuleNotFoundError:  # Direct ``python scripts/verify_setup.py`` execution.
    from prepare_tau2_dataset import SMOKE_IDS

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
    assert set(SMOKE_IDS) <= set(split_local["rl_train"])
    assert not set(SMOKE_IDS) & set(split_local["internal_dev"])

    official_actions = {
        str(task["id"]): {
            _canonical(action)
            for action in (task.get("evaluation_criteria") or {}).get("actions", [])
        }
        for task in tasks
    }
    selected_count = 0
    action_ids: dict[str, set[str]] = {}
    for name, expected_ids in (("train", train_ids), ("test", test_ids)):
        rows = _read(
            project_root / f"data/annotations/airline_required_actions.{name}.v1.json"
        )
        assert {str(row["id"]) for row in rows} == expected_ids
        for row in rows:
            action_ids[str(row["id"])] = {
                str(action["action_id"]) for action in row["actions"]
            }
            for action in row["actions"]:
                assert _canonical(action) in official_actions[str(row["id"])]
                selected_count += 1

        dependencies = _read(
            project_root
            / f"data/annotations/airline_action_dependencies.{name}.v1.json"
        )
        for row in dependencies:
            task_id = str(row["task_id"])
            assert task_id in expected_ids
            for predecessor, successor in row["dependencies"]:
                assert predecessor in action_ids[task_id]
                assert successor in action_ids[task_id]
                assert predecessor != successor

        policy_rows = _read(
            project_root
            / f"data/annotations/airline_mandatory_policy_rules.{name}.v1.json"
        )
        assert {str(row["task_id"]) for row in policy_rows} == expected_ids
        for row in policy_rows:
            assert row["deterministic_rules"] == [
                "confirmation_before_database_write",
                "one_tool_call_per_assistant_turn",
            ]
            assert row["judge_checks"] == policy_checks(str(row["task_id"]))

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

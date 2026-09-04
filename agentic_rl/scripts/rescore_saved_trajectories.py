"""Offline-rescore saved trajectories without calling Tau2 or an LLM."""

from __future__ import annotations

import argparse
from pathlib import Path

from tau2_agentic_rl.annotations import load_task_mapping
from tau2_agentic_rl.config import load_yaml
from tau2_agentic_rl.reward.required_actions import (
    load_action_dependencies,
    load_required_actions,
)
from tau2_agentic_rl.reward.score import build_reward_config, score_trajectory
from tau2_agentic_rl.schemas import TrajectoryRecord
from tau2_agentic_rl.storage import TrajectoryStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reward-version", default="v1-rescored")
    args = parser.parse_args()

    config = load_yaml(args.config)
    root = args.config.resolve().parents[2]
    annotations = config["annotations"]
    required = load_required_actions(root / annotations["required_actions"])
    dependencies = load_action_dependencies(root / annotations["action_dependencies"])
    transfer = load_task_mapping(root / annotations["transfer_rules"])
    reward_config = build_reward_config(config)
    store = TrajectoryStore(args.output_dir)

    count = 0
    for path in sorted(args.records_dir.glob("*.json")):
        record = TrajectoryRecord.model_validate_json(path.read_text(encoding="utf-8"))
        if record.official_scores is None or record.judge_result is None:
            raise ValueError(f"record lacks frozen scorer inputs: {path}")
        reward = score_trajectory(
            events=record.tool_events,
            messages=record.messages,
            assistant_turns=record.assistant_turns,
            required_actions=required[record.task_id],
            official=record.official_scores,
            judge=record.judge_result,
            transfer_rule=transfer[record.task_id],
            action_dependencies=dependencies.get(record.task_id, []),
            config=reward_config,
        )
        updated = record.model_copy(
            update={"reward_version": args.reward_version, "custom_reward": reward}
        )
        store.save(updated)
        count += 1
    print(f"rescored {count} trajectories into {args.output_dir}")


if __name__ == "__main__":
    main()

"""Offline-rescore saved trajectories without calling Tau2 or an LLM."""

from __future__ import annotations

import argparse
from pathlib import Path

from tau2_agentic_rl.annotations import load_task_mapping
from tau2_agentic_rl.config import load_yaml
from tau2_agentic_rl.judge.prompts import rubric_fingerprint
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
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--reward-version", default="v1-rescored")
    args = parser.parse_args()

    config = load_yaml(args.config)
    root = args.project_root.resolve()
    if args.records_dir.resolve() == args.output_dir.resolve():
        raise ValueError("offline rescoring must not overwrite source trajectories")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("offline rescoring requires a new empty output directory")
    annotations = config["annotations"]
    required = load_required_actions(root / annotations["required_actions"])
    dependencies = load_action_dependencies(root / annotations["action_dependencies"])
    transfer = load_task_mapping(root / annotations["transfer_rules"])
    semantic = load_task_mapping(root / annotations["semantic_checks"])
    policy = load_task_mapping(root / annotations["policy_rules"])
    reward_config = build_reward_config(config)
    store = TrajectoryStore(args.output_dir, attach_evaluation_identity=False)

    count = 0
    for path in sorted(args.records_dir.glob("*.json")):
        record = TrajectoryRecord.model_validate_json(path.read_text(encoding="utf-8"))
        if record.official_scores is None or record.judge_result is None:
            raise ValueError(f"record lacks frozen scorer inputs: {path}")
        expected_rubric = rubric_fingerprint(
            semantic[record.task_id].get("semantic_checks", []),
            policy[record.task_id].get("judge_checks", []),
            transfer[record.task_id],
        )
        if record.metadata.get("judge_rubric_sha256") != expected_rubric:
            raise ValueError(
                f"Judge rubric changed or was not fingerprinted; re-judge before rescoring: {path}"
            )
        if record.environment_transcript is None:
            raise ValueError(
                f"record lacks the delivered environment transcript: {path}"
            )
        reward = score_trajectory(
            events=record.tool_events,
            messages=record.environment_transcript,
            assistant_turns=record.assistant_turns,
            required_actions=required[record.task_id],
            official=record.official_scores,
            judge=record.judge_result,
            transfer_rule=transfer[record.task_id],
            action_dependencies=dependencies.get(record.task_id, []),
            config=reward_config,
        )
        metadata = dict(record.metadata)
        previous_manifest = metadata.pop("evaluation_manifest_id", None)
        if previous_manifest:
            metadata["rescored_from_evaluation_manifest_id"] = previous_manifest
        updated = record.model_copy(
            update={
                "reward_version": args.reward_version,
                "custom_reward": reward,
                "metadata": metadata,
            }
        )
        store.save(updated)
        count += 1
    print(f"rescored {count} trajectories into {args.output_dir}")


if __name__ == "__main__":
    main()

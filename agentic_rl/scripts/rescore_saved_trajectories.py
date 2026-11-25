"""Offline-rescore saved trajectories without calling Tau2 or an LLM."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from tau2_agentic_rl.agent_policy import load_agent_system_prompt, prompt_sha256
from tau2_agentic_rl.annotations import load_task_judge_guidance, load_task_mapping
from tau2_agentic_rl.config import load_yaml
from tau2_agentic_rl.full_official_group import OPTION as FULL_GROUP_OPTION, terminal_only_reward
from tau2_agentic_rl.judge.prompts import rubric_fingerprint
from tau2_agentic_rl.judge.routed import MODE as ROUTED_MODE
from tau2_agentic_rl.judge.routed import record_verdict, routed_fingerprint, scored_trace
from tau2_agentic_rl.procredit_runtime import credit_from_record
from tau2_agentic_rl.reward.truncation import deterministic_zero_reward, zero_truncated
from tau2_agentic_rl.reward.required_actions import (
    load_action_dependencies,
    load_required_actions,
)
from tau2_agentic_rl.reward.score import build_reward_config, score_trajectory
from tau2_agentic_rl.reward.simple_turn import (
    build_label_trace,
    enabled,
    validate_record_labels,
)
from tau2_agentic_rl.schemas import TrajectoryRecord
from tau2_agentic_rl.storage import TrajectoryStore
from tau2_agentic_rl.versions import sha256_json


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
    expected_prompt_hash = prompt_sha256(load_agent_system_prompt(config, root))
    if args.records_dir.resolve() == args.output_dir.resolve():
        raise ValueError("offline rescoring must not overwrite source trajectories")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("offline rescoring requires a new empty output directory")
    annotations = config["annotations"]
    required = load_required_actions(root / annotations["required_actions"])
    dependencies = load_action_dependencies(root / annotations["action_dependencies"])
    transfer = load_task_mapping(root / annotations["transfer_rules"])
    semantic = load_task_mapping(root / annotations["semantic_checks"])
    task_guidance = load_task_judge_guidance(root, annotations, set(semantic))
    policy = load_task_mapping(root / annotations["policy_rules"])
    routed = config.get("judge", {}).get("mode") == ROUTED_MODE
    exclude_queries = config.get("judge", {}).get("exclude_pure_queries", False)
    routes = load_task_mapping(root / annotations["judge_routes"]) if routed else {}
    reward_config = build_reward_config(config)
    store = TrajectoryStore(args.output_dir, attach_evaluation_identity=False)

    count = 0
    for path in sorted(args.records_dir.glob("*.json")):
        record = TrajectoryRecord.model_validate_json(path.read_text(encoding="utf-8"))
        if record.metadata.get("agent_system_prompt_sha256") != expected_prompt_hash:
            raise ValueError(
                f"record was not generated with the bound SFT system prompt: {path}"
            )
        source_mode = (record.scoring_inputs or {}).get("reward_project_config", {}).get("judge", {}).get("mode")
        if (source_mode == ROUTED_MODE) != routed:
            raise ValueError("Judge mode changed; frozen decisions cannot be reused across protocols")
        source_judge = (record.scoring_inputs or {}).get("reward_project_config", {}).get("judge", {})
        if source_judge.get("exclude_pure_queries", False) != exclude_queries:
            raise ValueError("Judge target filter changed; frozen decisions require their original configuration")
        if routed and record.scoring_inputs.get("routing_rule") != routes[record.task_id]:
            raise ValueError("frozen task routing rule changed")
        truncated = routed and zero_truncated(config, record.termination_reason)
        group_only = "official_group" in (record.scoring_inputs or {})
        if group_only and (truncated or not config.get("reward", {}).get(FULL_GROUP_OPTION)):
            raise ValueError("cannot use frozen terminal-only scores under a different branch")
        judge = record_verdict(record) if routed and not (truncated or group_only) else record.judge_result
        if record.official_scores is None or (judge is None and not (truncated or group_only)):
            raise ValueError(f"record lacks frozen scorer inputs: {path}")
        expected_rubric = rubric_fingerprint(
            semantic[record.task_id].get("semantic_checks", []),
            [] if routed else policy[record.task_id].get("judge_checks", []),
            transfer[record.task_id],
            require_policy_attribution=reward_config.credit_version == "procredit-turn-v4",
            transfer_validation_version=reward_config.transfer_validation_version,
            semantic_checks_only=semantic[record.task_id].get("semantic_checks_only", False),
            task_guidance=task_guidance.get(record.task_id),
        )
        if routed:
            expected_rubric = routed_fingerprint(expected_rubric, routes[record.task_id], exclude_pure_queries=exclude_queries)
        if record.metadata.get("judge_rubric_sha256") != expected_rubric:
            raise ValueError(
                f"Judge rubric changed or was not fingerprinted; re-judge before rescoring: {path}"
            )
        if record.environment_transcript is None:
            raise ValueError(
                f"record lacks the delivered environment transcript: {path}"
            )
        if truncated or group_only:
            credit_from_record(record)
        elif enabled(config):
            # This command has no model transport: never invent missing labels
            # when an old trajectory is rescored under the new default config.
            validate_record_labels(record)
        if group_only:
            from tau2_agentic_rl.reward.progress import validate_progress_inputs
            validate_progress_inputs(record.progress_trace, required_actions=required[record.task_id],
                                     dependencies=dependencies.get(record.task_id, []),
                                     transfer_rule=transfer[record.task_id])
            reward = terminal_only_reward(record.progress_trace, record.assistant_turns)
        elif truncated:
            reward = deterministic_zero_reward(
                termination_reason=record.termination_reason, project=config, official=record.official_scores,
            )
        else:
            reward = score_trajectory(
                events=record.tool_events,
                messages=record.environment_transcript,
                assistant_turns=record.assistant_turns,
                required_actions=required[record.task_id],
                official=record.official_scores,
                judge=judge,
                transfer_rule=transfer[record.task_id],
                action_dependencies=dependencies.get(record.task_id, []),
                termination_reason=record.termination_reason,
                config=reward_config,
                progress_trace=record.progress_trace,
            )
        metadata = dict(record.metadata)
        scoring_inputs = record.scoring_inputs
        if reward_config.credit_version in {"procredit-turn-v3", "procredit-turn-v4"}:
            source_fingerprint = sha256_json(scoring_inputs)
            if not scoring_inputs or source_fingerprint != metadata.get("scoring_inputs_sha256"):
                raise ValueError("v3 rescoring requires intact frozen scoring inputs")
            scoring_inputs = deepcopy(scoring_inputs)
            metadata["rescored_from_scoring_inputs_sha256"] = source_fingerprint
            metadata["rescored_from_reward_version"] = record.reward_version
            scoring_inputs["reward_project_config"] = {
                key: deepcopy(config.get(key, {})) for key in ("reward", "credit", "rollout")
            }
            if source_mode is not None:
                scoring_inputs["reward_project_config"]["judge"] = dict(source_judge)
            metadata["scoring_inputs_sha256"] = sha256_json(scoring_inputs)
        previous_manifest = metadata.pop("evaluation_manifest_id", None)
        if previous_manifest:
            metadata["rescored_from_evaluation_manifest_id"] = previous_manifest
        updated = record.model_copy(
            update={
                "reward_version": args.reward_version,
                "custom_reward": reward,
                "metadata": metadata,
                "scoring_inputs": scoring_inputs,
            }
        )
        if truncated or group_only:
            credit_from_record(updated)
        elif enabled(config):
            previous = record.metadata["simple_turn_labels"]
            if routed:
                updated.metadata["simple_turn_labels"] = scored_trace(
                    scoring_inputs["simple_turn_inputs"], record.tool_events, reward.details["policy_credit"],
                    tuple(metadata.get(k) for k in ("judge_raw", "judge_prompt_hash", "judge_cache_key")),
                )
            else:
                updated.metadata["simple_turn_labels"] = {
                    **previous,
                    **build_label_trace(
                        scoring_inputs["simple_turn_inputs"], previous["decisions"],
                        record.tool_events, reward.details["policy_credit"],
                    ),
                }
            validate_record_labels(updated)
        store.save(updated)
        count += 1
    print(f"rescored {count} trajectories into {args.output_dir}")


if __name__ == "__main__":
    main()

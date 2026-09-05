"""Run frozen Tau2 Airline evaluation without optimizer updates."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

try:
    from scripts.prepare_tau2_dataset import _row, _write_parquet
    from scripts.summarize_evaluation import summarize
    from scripts.train_airline_grpo import (
        TAU2_COMMIT,
        VERL_COMMIT,
        _require_env,
        _require_exact_checkout,
        _safe_run_name,
    )
except ModuleNotFoundError:  # Direct python scripts/evaluate_airline.py.
    from prepare_tau2_dataset import _row, _write_parquet
    from summarize_evaluation import summarize
    from train_airline_grpo import (
        TAU2_COMMIT,
        VERL_COMMIT,
        _require_env,
        _require_exact_checkout,
        _safe_run_name,
    )

from tau2_agentic_rl.base_identity import validate_adapter_base
from tau2_agentic_rl.config import load_runtime_config
from tau2_agentic_rl.evaluation import (
    evaluation_coverage,
    evaluation_lock,
    fingerprint_directory,
    initialize_evaluation,
)
from tau2_agentic_rl.judge.client import DeepSeekJudge, JudgeConfig
from tau2_agentic_rl.schemas import TrajectoryRecord
from tau2_agentic_rl.scoring_retry import retry_scoring
from tau2_agentic_rl.storage import TrajectoryStore
from tau2_agentic_rl.versions import sha256_file, sha256_json


def build_evaluation_command(
    args, *, project_root, data_file, run_root, project, identity
):
    agent_config = project_root / "configs" / "rl" / "agent_loop_v1.yaml"
    command = [
        sys.executable,
        "-m",
        "tau2_agentic_rl.verl_entrypoint",
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=false",
        "algorithm.filter_groups.enable=false",
        f"data.train_files={data_file}",
        f"data.val_files={data_file}",
        "data.train_batch_size=1",
        "data.max_prompt_length=8192",
        "data.max_response_length=16384",
        "data.return_raw_chat=true",
        "data.filter_overlong_prompts=false",
        "data.truncation=error",
        "data.continuous_token.enable=false",
        f"actor_rollout_ref.model.path={args.model_path}",
        "actor_rollout_ref.model.use_remove_padding=true",
        "actor_rollout_ref.actor.use_dynamic_bsz=false",
        "actor_rollout_ref.actor.ppo_mini_batch_size=1",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.40",
        "actor_rollout_ref.rollout.max_model_len=16384",
        "actor_rollout_ref.rollout.calculate_log_probs=true",
        f"actor_rollout_ref.rollout.agent.num_workers={project['rollout']['agent_worker_count']}",
        f"actor_rollout_ref.rollout.max_num_seqs={project['rollout']['vllm_max_num_seqs']}",
        "actor_rollout_ref.rollout.agent.default_agent_loop=tau2_airline",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={agent_config}",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={identity['temperature']}",
        f"actor_rollout_ref.rollout.val_kwargs.top_p={identity['top_p']}",
        f"actor_rollout_ref.rollout.val_kwargs.top_k={identity['top_k']}",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=true",
        "actor_rollout_ref.rollout.val_kwargs.n=1",
        f"actor_rollout_ref.rollout.seed={args.seed}",
        f"data.seed={args.seed}",
        "trainer.use_v1=true",
        "trainer.v1.trainer_mode=sync",
        "trainer.val_only=true",
        "trainer.val_before_train=true",
        "trainer.resume_mode=disable",
        f"trainer.default_local_dir={run_root / 'unused_checkpoints'}",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.logger=[console]",
        "trainer.project_name=tau2_airline_agentic_rl",
        f"trainer.experiment_name={args.tag}",
    ]
    if args.lora_adapter:
        adapter_config = json.loads(
            (args.lora_adapter / "adapter_config.json").read_text(encoding="utf-8")
        )
        command.extend(
            [
                f"actor_rollout_ref.model.lora_rank={adapter_config['r']}",
                f"actor_rollout_ref.model.lora_alpha={adapter_config['lora_alpha']}",
                "actor_rollout_ref.model.target_modules=all-linear",
                f"actor_rollout_ref.model.lora_adapter_path={args.lora_adapter}",
                "actor_rollout_ref.rollout.load_format=safetensors",
            ]
        )
    command.extend(args.extra)
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau2-root", type=Path, default=os.environ.get("TAU2_ROOT"))
    parser.add_argument("--verl-root", type=Path, default=os.environ.get("VERL_ROOT"))
    parser.add_argument("--model-path", default=os.environ.get("MERGED_SFT_MODEL"))
    parser.add_argument("--lora-adapter", type=Path)
    parser.add_argument(
        "--split",
        choices=("internal_dev", "official_train", "official_test"),
        default="official_test",
    )
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--tag", default="frozen_test")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-refill-rounds", type=int, default=2)
    args = parser.parse_args()

    if args.tau2_root is None or args.verl_root is None or not args.model_path:
        raise RuntimeError("set TAU2_ROOT, VERL_ROOT, and MERGED_SFT_MODEL/model-path")
    _require_exact_checkout(args.tau2_root, TAU2_COMMIT, "Tau2")
    _require_exact_checkout(args.verl_root, VERL_COMMIT, "veRL")
    args.model_path = str(Path(args.model_path).resolve())
    if not (Path(args.model_path) / "config.json").is_file() or not any(
        Path(args.model_path).glob("*.safetensors")
    ):
        raise FileNotFoundError("evaluation requires a complete local merged model")
    if args.samples < 4 or (args.split == "official_test" and args.samples != 4):
        raise ValueError(
            "official test requires exactly 4 samples; other splits need at least 4"
        )
    if args.max_refill_rounds < 0:
        raise ValueError("max-refill-rounds must be nonnegative")
    for override in args.extra:
        if override.split("=", 1)[0] not in {
            "actor_rollout_ref.rollout.gpu_memory_utilization",
            "actor_rollout_ref.rollout.agent.num_workers",
        }:
            raise ValueError(
                "evaluation --extra only accepts memory utilization or worker count"
            )
    args.tag = _safe_run_name(args.tag)
    if args.lora_adapter is not None:
        args.lora_adapter = args.lora_adapter.resolve()
        required = [
            args.lora_adapter / "adapter_config.json",
            args.lora_adapter / "adapter_model.safetensors",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "incomplete RL PEFT adapter; export it with "
                "scripts/export_verl_lora.py: " + ", ".join(missing)
            )
        validate_adapter_base(args.lora_adapter, args.model_path)
    for name in (
        "DEEPSEEK_USER_MODEL",
        "DEEPSEEK_JUDGE_MODEL",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
    ):
        _require_env(name)

    project_root = Path(__file__).resolve().parents[1]
    test_mode = args.split == "official_test"
    config_path = (
        project_root
        / "configs"
        / (
            "evaluation/airline_eval_v1.yaml"
            if test_mode
            else "rl/airline_grpo_v1.yaml"
        )
    )
    run_root = project_root / "outputs/evaluations" / args.tag
    data_file = run_root / "pending_samples.parquet"

    os.environ["AGENTIC_RL_CONFIG"] = str(config_path)
    os.environ["TRAJECTORY_OUTPUT_DIR"] = f"outputs/evaluations/{args.tag}/trajectories"
    os.environ["JUDGE_CACHE_DIR"] = f"outputs/evaluations/{args.tag}/judge_cache"
    os.environ["USER_CACHE_DIR"] = f"outputs/evaluations/{args.tag}/user_cache"
    os.environ["AGENTIC_RL_PROJECT_ROOT"] = str(project_root)
    os.environ["MERGED_SFT_MODEL"] = args.model_path
    project = load_runtime_config(config_path)
    split_path = project_root / "data/splits/airline_internal_dev.v1.json"
    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    task_ids = (
        split_data[args.split]
        if args.split != "official_train"
        else sorted(split_data["rl_train"] + split_data["internal_dev"], key=int)
    )
    record_split = {
        "official_test": "test",
        "official_train": "train",
        "internal_dev": "internal_dev",
    }[args.split]
    if test_mode and len(task_ids) != 20:
        raise ValueError("frozen test must contain 20 tasks")
    identity = {
        "model_path": args.model_path,
        "model_files": fingerprint_directory(Path(args.model_path)),
        "adapter_path": str(args.lora_adapter) if args.lora_adapter else None,
        "adapter_sha256": sha256_json(fingerprint_directory(args.lora_adapter))
        if args.lora_adapter
        else None,
        "config_sha256": sha256_json(project),
        "annotation_sha256": sha256_json(
            {
                name: sha256_file(project_root / path)
                for name, path in project["annotations"].items()
            }
        ),
        "source_sha256": sha256_json(
            {
                str(p.relative_to(project_root)): sha256_file(p)
                for folder in ("src", "scripts")
                for p in sorted((project_root / folder).rglob("*.py"))
            }
        ),
        "split_sha256": sha256_file(split_path),
        "tau2_commit": TAU2_COMMIT,
        "verl_commit": VERL_COMMIT,
        "user_model": project["user_simulator"]["model"],
        "judge_model": project["judge"]["model"],
        "temperature": project["rollout"]["temperature"],
        "top_p": project["rollout"]["top_p"],
        "top_k": project["rollout"]["top_k"],
        "samples_per_task": args.samples,
        "task_ids": task_ids,
        "seed": args.seed,
        "split": args.split,
        "record_split": record_split,
        "extra": args.extra,
    }
    path_items = [
        str(project_root / "src"),
        str(args.tau2_root / "src"),
        str(args.verl_root),
    ]
    if os.environ.get("PYTHONPATH"):
        path_items.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(path_items)

    command = build_evaluation_command(
        args,
        project_root=project_root,
        data_file=data_file,
        run_root=run_root,
        project=project,
        identity=identity,
    )
    print(shlex.join(command))
    if not args.dry_run:
        manifest = initialize_evaluation(run_root, identity, resume=args.resume)
        os.environ["EVALUATION_MANIFEST_ID"] = manifest["manifest_id"]
        with evaluation_lock(run_root):
            runtime_path = run_root / "runtime_config.yaml"
            if not runtime_path.exists():
                runtime_path.write_text(
                    yaml.safe_dump(project, sort_keys=False), encoding="utf-8"
                )
            elif (
                sha256_json(yaml.safe_load(runtime_path.read_text(encoding="utf-8")))
                != identity["config_sha256"]
            ):
                raise ValueError(
                    "saved runtime config differs from evaluation manifest"
                )
            os.environ["AGENTIC_RL_CONFIG"] = str(runtime_path)
            judge_config = project["judge"]
            judge = DeepSeekJudge(
                JudgeConfig(
                    model=judge_config["model"],
                    provider=judge_config.get("provider", "DeepSeek"),
                    base_url=judge_config["base_url"],
                    max_retries=int(judge_config["max_retries"]),
                    cache_dir=str(project_root / project["outputs"]["judge_cache"]),
                )
            )
            store = TrajectoryStore(run_root / "trajectories")
            for refill in range(args.max_refill_rounds + 1):
                coverage = evaluation_coverage(run_root / "trajectories", manifest)
                for row in coverage["scoring_pending_records"]:
                    asyncio.run(
                        retry_scoring(
                            TrajectoryRecord.model_validate(row), judge, store
                        )
                    )
                coverage = evaluation_coverage(run_root / "trajectories", manifest)
                if coverage["complete"]:
                    break
                if not coverage["missing_slots"]:
                    continue  # Scoring-only failures must never cause a fresh interaction.
                rows = []
                for item in coverage["missing_slots"]:
                    task, slot = item["task_id"], item["sample_index"]
                    row = _row(
                        task, record_split, args.seed + int(task) * args.samples + slot
                    )
                    row["extra_info"]["evaluation_sample_index"] = slot
                    rows.append(row)
                _write_parquet(rows, data_file)
                result = subprocess.run(command, check=False, cwd=args.verl_root)
                print(f"evaluation round {refill}: returncode={result.returncode}")
            result = summarize(run_root / "trajectories")
            (run_root / "summary.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

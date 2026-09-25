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

from tau2_agentic_rl.agent_policy import load_agent_system_prompt
from tau2_agentic_rl.base_identity import validate_adapter_base
from tau2_agentic_rl.config import expand_env, load_yaml
from tau2_agentic_rl.evaluation import (
    evaluation_coverage,
    evaluation_lock,
    fingerprint_directory,
    initialize_evaluation,
)
from tau2_agentic_rl.judge.client import DeepSeekJudge, JudgeConfig
from tau2_agentic_rl.pass_metrics import (
    OFFICIAL_EVALUATION_PROTOCOL_SAMPLES,
    TAU2_OFFICIAL_AGENT_TEMPERATURE,
    TAU2_OFFICIAL_MAX_ERRORS,
    TAU2_OFFICIAL_MAX_STEPS,
    TAU2_OFFICIAL_SEED,
    TAU2_OFFICIAL_USER_TEMPERATURE,
    official_trial_seeds,
    validate_official_test_ids,
)
from tau2_agentic_rl.schemas import TrajectoryRecord
from tau2_agentic_rl.scoring_retry import retry_scoring_batch
from tau2_agentic_rl.storage import TrajectoryStore
from tau2_agentic_rl.user_simulation import (
    build_user_sim_judge,
    check_user_simulation,
    filter_enabled,
    replacement_seed,
)
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
        f"data.max_prompt_length={project['rollout']['initial_prompt_max_tokens']}",
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


def parse_args(argv=None):
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
    parser.add_argument(
        "--protocol",
        choices=tuple(OFFICIAL_EVALUATION_PROTOCOL_SAMPLES),
        default="pass1_pass4",
        help="Official-test protocol: 20x1 pass^1 or 20x4 pass^1/pass^4",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Legacy explicit sample count; must agree with --protocol on official_test",
    )
    parser.add_argument("--tag", type=_safe_run_name, default="frozen_test")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra", action="append", default=[])
    parser.add_argument("--seed", type=int, default=TAU2_OFFICIAL_SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-refill-rounds", type=int, default=3)
    parser.add_argument(
        "--reward-judge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Agent reward Judge/custom metrics (default: official pass only).",
    )
    parser.add_argument(
        "--user-sim-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable optional User Simulator compliance screening (default: disabled)."
        ),
    )
    return parser.parse_args(argv)


def resolve_evaluation_samples(args) -> int:
    if args.split == "official_test" and args.seed != TAU2_OFFICIAL_SEED:
        raise ValueError("official_test requires the pinned Tau2 seed=300")
    if args.split == "official_test":
        expected = OFFICIAL_EVALUATION_PROTOCOL_SAMPLES[args.protocol]
        if args.samples is not None and args.samples != expected:
            raise ValueError(
                f"--samples {args.samples} conflicts with --protocol {args.protocol} "
                f"(requires {expected})"
            )
        return expected
    if args.protocol != "pass1_pass4":
        raise ValueError("--protocol pass1 is only valid with --split official_test")
    samples = 4 if args.samples is None else args.samples
    if samples < 4:
        raise ValueError("non-test evaluation needs at least four samples")
    return samples


def configure_evaluation_runtime(
    project: dict, *, reward_judge: bool, user_sim_filter: bool
) -> dict:
    """Apply pinned Tau2 CLI defaults without changing the chosen user model."""
    rollout = project["rollout"]
    rollout.pop("max_soft_turns", None)
    rollout["max_hard_turns"] = None
    rollout["tau2_max_steps"] = TAU2_OFFICIAL_MAX_STEPS
    rollout["tau2_max_errors"] = TAU2_OFFICIAL_MAX_ERRORS
    rollout["temperature"] = TAU2_OFFICIAL_AGENT_TEMPERATURE
    max_context_tokens = int(rollout["max_context_length"])
    rollout["initial_prompt_max_tokens"] = max_context_tokens
    rollout["reserved_observation_tokens"] = 0
    rollout["reserved_template_tokens"] = 0
    rollout["min_final_response_tokens"] = 1
    rollout["per_turn_max_new_tokens"] = max_context_tokens
    rollout["observation_content_max_tokens"] = max_context_tokens
    project["user_simulator"]["temperature"] = TAU2_OFFICIAL_USER_TEMPERATURE
    if reward_judge:
        project["judge"]["enabled"] = True
    else:
        project["judge"] = {"enabled": False}
        project.pop("reward", None)
    if user_sim_filter:
        project["user_sim_filter"]["enabled"] = True
    else:
        project["user_sim_filter"] = {"enabled": False}
    return project


def main() -> None:
    args = parse_args()

    if args.tau2_root is None or args.verl_root is None or not args.model_path:
        raise RuntimeError("set TAU2_ROOT, VERL_ROOT, and MERGED_SFT_MODEL/model-path")
    _require_exact_checkout(args.tau2_root, TAU2_COMMIT, "Tau2")
    _require_exact_checkout(args.verl_root, VERL_COMMIT, "veRL")
    args.model_path = str(Path(args.model_path).resolve())
    if not (Path(args.model_path) / "config.json").is_file() or not any(
        Path(args.model_path).glob("*.safetensors")
    ):
        raise FileNotFoundError("evaluation requires a complete local merged model")
    args.samples = resolve_evaluation_samples(args)
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
    os.environ["USER_SIM_JUDGE_CACHE_DIR"] = (
        f"outputs/evaluations/{args.tag}/user_sim_judge_cache"
    )
    os.environ["AGENTIC_RL_PROJECT_ROOT"] = str(project_root)
    os.environ["MERGED_SFT_MODEL"] = args.model_path
    project = load_yaml(config_path)
    project = configure_evaluation_runtime(
        project,
        reward_judge=args.reward_judge,
        user_sim_filter=args.user_sim_filter,
    )
    for name in (
        "DEEPSEEK_USER_MODEL",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
    ):
        _require_env(name)
    if filter_enabled(project):
        _require_env("DEEPSEEK_USER_SIM_JUDGE_MODEL")
    if args.reward_judge:
        _require_env("DEEPSEEK_JUDGE_MODEL")
    project = expand_env(project)
    load_agent_system_prompt(project, project_root)

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
    if test_mode:
        validate_official_test_ids(task_ids, TAU2_COMMIT)
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
        "judge_model": project["judge"].get("model"),
        "reward_judge_enabled": args.reward_judge,
        "user_sim_filter_enabled": filter_enabled(project),
        "user_sim_filter": project.get("user_sim_filter", {}),
        "evaluation_standard": (
            "tau2_nonofficial_user_sim_filtered"
            if filter_enabled(project)
            else "tau2_official"
        ),
        "tau2_max_steps": project["rollout"]["tau2_max_steps"],
        "tau2_max_errors": project["rollout"]["tau2_max_errors"],
        "assistant_turn_limit": project["rollout"].get("max_hard_turns"),
        "temperature": project["rollout"]["temperature"],
        "user_temperature": project["user_simulator"]["temperature"],
        "top_p": project["rollout"]["top_p"],
        "top_k": project["rollout"]["top_k"],
        "samples_per_task": args.samples,
        **({"evaluation_protocol": args.protocol} if test_mode else {}),
        "task_ids": task_ids,
        "seed": args.seed,
        "trial_seeds": official_trial_seeds(args.seed, args.samples),
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
            judge = (
                DeepSeekJudge(
                    JudgeConfig(
                        model=judge_config["model"],
                        provider=judge_config.get("provider", "DeepSeek"),
                        base_url=judge_config["base_url"],
                        max_retries=int(judge_config["max_retries"]),
                        cache_dir=str(project_root / project["outputs"]["judge_cache"]),
                    )
                )
                if args.reward_judge
                else None
            )
            user_sim_judge = (
                build_user_sim_judge(project, project_root)
                if filter_enabled(project)
                else None
            )
            store = TrajectoryStore(run_root / "trajectories")
            for refill in range(args.max_refill_rounds + 1):
                coverage = evaluation_coverage(run_root / "trajectories", manifest)

                async def retry_user_checks():
                    pending = iter(coverage["user_sim_pending_records"])

                    async def worker():
                        for row in pending:
                            try:
                                await check_user_simulation(
                                    TrajectoryRecord.model_validate(row),
                                    user_sim_judge,
                                    store,
                                )
                            except Exception as exc:
                                print(
                                    f"user-simulator check still pending: {row['trajectory_id']}: {type(exc).__name__}"
                                )

                    async with asyncio.TaskGroup() as group:
                        for _ in range(
                            min(4, len(coverage["user_sim_pending_records"]))
                        ):
                            group.create_task(worker())

                asyncio.run(retry_user_checks())
                coverage = evaluation_coverage(run_root / "trajectories", manifest)
                asyncio.run(
                    retry_scoring_batch(
                        [
                            TrajectoryRecord.model_validate(row)
                            for row in coverage["scoring_pending_records"]
                        ],
                        judge,
                        store,
                    )
                )
                coverage = evaluation_coverage(run_root / "trajectories", manifest)
                if coverage["complete"]:
                    break
                if not coverage["missing_slots"]:
                    continue  # Scoring-only failures must never cause a fresh interaction.
                rows = []
                trial_seeds = official_trial_seeds(args.seed, args.samples)
                for item in coverage["missing_slots"]:
                    task, slot = item["task_id"], item["sample_index"]
                    attempt = item.get("user_sim_attempt", 0)
                    if (
                        filter_enabled(project)
                        and attempt > project["user_sim_filter"]["max_resamples"]
                    ):
                        raise RuntimeError(
                            "user-simulator replacement cap reached; evaluation remains incomplete"
                        )
                    row = _row(
                        task,
                        record_split,
                        replacement_seed(trial_seeds[slot], attempt),
                    )
                    row["extra_info"]["evaluation_sample_index"] = slot
                    row["extra_info"]["user_sim_attempt"] = attempt
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

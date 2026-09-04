"""Run frozen Tau2 Airline evaluation without optimizer updates."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from train_airline_grpo import (
    TAU2_COMMIT,
    VERL_COMMIT,
    _require_env,
    _require_exact_checkout,
)


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
    args = parser.parse_args()

    if args.tau2_root is None or args.verl_root is None or not args.model_path:
        raise RuntimeError("set TAU2_ROOT, VERL_ROOT, and MERGED_SFT_MODEL/model-path")
    _require_exact_checkout(args.tau2_root, TAU2_COMMIT, "Tau2")
    _require_exact_checkout(args.verl_root, VERL_COMMIT, "veRL")
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
    data_file = project_root / "data" / "parquet" / f"airline_{args.split}.parquet"
    if not data_file.exists():
        raise FileNotFoundError("run scripts/prepare_tau2_dataset.py first")

    os.environ["AGENTIC_RL_CONFIG"] = str(config_path)
    os.environ["TRAJECTORY_OUTPUT_DIR"] = f"outputs/evaluations/{args.tag}/trajectories"
    os.environ["JUDGE_CACHE_DIR"] = f"outputs/evaluations/{args.tag}/judge_cache"
    os.environ["USER_CACHE_DIR"] = f"outputs/evaluations/{args.tag}/user_cache"
    path_items = [
        str(project_root / "src"),
        str(args.tau2_root / "src"),
        str(args.verl_root),
    ]
    if os.environ.get("PYTHONPATH"):
        path_items.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(path_items)

    agent_config = project_root / "configs" / "rl" / "agent_loop_v1.yaml"
    command = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=false",
        "algorithm.filter_groups.enable=false",
        f"data.train_files={data_file}",
        f"data.val_files={data_file}",
        "data.train_batch_size=1",
        "data.max_prompt_length=8192",
        "data.max_response_length=16384",
        "data.return_raw_chat=true",
        "data.filter_overlong_prompts=true",
        "data.truncation=error",
        "data.continuous_token.enable=false",
        f"actor_rollout_ref.model.path={args.model_path}",
        "actor_rollout_ref.model.use_remove_padding=true",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.40",
        "actor_rollout_ref.rollout.max_model_len=16384",
        "actor_rollout_ref.rollout.calculate_log_probs=true",
        "actor_rollout_ref.rollout.agent.num_workers=2",
        "actor_rollout_ref.rollout.agent.default_agent_loop=tau2_airline",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={agent_config}",
        "actor_rollout_ref.rollout.val_kwargs.temperature=1.0",
        "actor_rollout_ref.rollout.val_kwargs.top_p=1.0",
        "actor_rollout_ref.rollout.val_kwargs.top_k=-1",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=true",
        f"actor_rollout_ref.rollout.val_kwargs.n={args.samples}",
        "trainer.use_v1=true",
        "trainer.v1.trainer_mode=sync",
        "trainer.val_only=true",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.logger=[console]",
        "trainer.project_name=tau2_airline_agentic_rl",
        f"trainer.experiment_name={args.tag}",
    ]
    if args.lora_adapter:
        command.extend(
            [
                "actor_rollout_ref.model.lora_rank=32",
                "actor_rollout_ref.model.lora_alpha=64",
                "actor_rollout_ref.model.target_modules=all-linear",
                f"actor_rollout_ref.model.lora_adapter_path={args.lora_adapter}",
                "actor_rollout_ref.rollout.load_format=safetensors",
            ]
        )
    command.extend(args.extra)
    print(shlex.join(command))
    if not args.dry_run:
        subprocess.run(command, check=True, cwd=args.verl_root)


if __name__ == "__main__":
    main()

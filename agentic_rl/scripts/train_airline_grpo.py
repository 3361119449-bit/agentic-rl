"""Validated launcher for the pinned veRL Tau2 Airline GRPO run."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

TAU2_COMMIT = "a2c024725189473d2d7cea3a5cfdbcc67478e41f"
VERL_COMMIT = "483b8a009ba3a97563edee3a19887e4862b8094a"


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _require_exact_checkout(root: Path, expected: str, label: str) -> None:
    actual = _git_commit(root)
    if actual != expected:
        raise RuntimeError(
            f"{label} commit mismatch: expected {expected}, got {actual}"
        )


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value or value.startswith("FIX_EXACT_"):
        raise RuntimeError(f"set {name} to an exact value before launching")
    return value


def build_command(
    *,
    project_root: Path,
    model_path: str,
    train_file: Path,
    val_file: Path,
    total_epochs: int,
    extra: list[str],
) -> list[str]:
    """Translate the plan to veRL v0.9.0 Hydra overrides."""
    agent_config = project_root / "configs" / "rl" / "agent_loop_v1.yaml"
    output_root = project_root / "outputs"
    return [
        sys.executable,
        "-m",
        "tau2_agentic_rl.verl_entrypoint",
        "algorithm.adv_estimator=grpo",
        "algorithm.norm_adv_by_std_in_grpo=true",
        "algorithm.use_kl_in_reward=false",
        "algorithm.kl_ctrl.kl_coef=0.0",
        "algorithm.rollout_correction.bypass_mode=true",
        "algorithm.rollout_correction.loss_type=ppo_clip",
        "algorithm.filter_groups.enable=true",
        "algorithm.filter_groups.metric=train_reward",
        "algorithm.filter_groups.max_num_gen_batches=3",
        "algorithm.filter_groups.max_inflight_gen_batches=1",
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        "data.train_batch_size=4",
        "data.max_prompt_length=8192",
        "data.max_response_length=16384",
        "data.return_raw_chat=true",
        "data.filter_overlong_prompts=true",
        "data.truncation=error",
        "data.continuous_token.enable=false",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.lora_rank=32",
        "actor_rollout_ref.model.lora_alpha=64",
        "actor_rollout_ref.model.target_modules=all-linear",
        "actor_rollout_ref.model.use_remove_padding=true",
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",
        "actor_rollout_ref.actor.optim.lr=5e-6",
        # veRL v0.9.0 interprets this field in prompt groups. Four prompts
        # times rollout.n=8 gives the planned 32-trajectory mini-batch.
        "actor_rollout_ref.actor.ppo_mini_batch_size=4",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.ppo_epochs=2",
        "actor_rollout_ref.actor.use_dynamic_bsz=true",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768",
        "actor_rollout_ref.actor.clip_ratio_low=0.20",
        "actor_rollout_ref.actor.clip_ratio_high=0.28",
        "actor_rollout_ref.actor.clip_ratio_c=10.0",
        "actor_rollout_ref.actor.loss_agg_mode=token-mean",
        "actor_rollout_ref.actor.entropy_coeff=0.0",
        "actor_rollout_ref.actor.grad_clip=1.0",
        "actor_rollout_ref.actor.use_kl_loss=false",
        "actor_rollout_ref.actor.kl_loss_coef=0.0",
        "actor_rollout_ref.actor.fsdp_config.param_offload=false",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.40",
        "actor_rollout_ref.rollout.n=8",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.top_k=-1",
        "actor_rollout_ref.rollout.calculate_log_probs=true",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.max_model_len=16384",
        "actor_rollout_ref.rollout.load_format=safetensors",
        "actor_rollout_ref.rollout.layered_summon=true",
        "actor_rollout_ref.rollout.agent.num_workers=2",
        "actor_rollout_ref.rollout.agent.default_agent_loop=tau2_airline",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={agent_config}",
        "trainer.use_v1=true",
        "trainer.v1.trainer_mode=sync",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.critic_warmup=0",
        "trainer.val_before_train=true",
        "trainer.logger=[console,wandb]",
        "trainer.project_name=tau2_airline_agentic_rl",
        "trainer.experiment_name=qwen3_4b_airline_grpo_v1",
        f"trainer.default_local_dir={output_root / 'checkpoints'}",
        f"trainer.rollout_data_dir={output_root / 'verl_rollouts'}",
        "trainer.save_freq=10",
        "trainer.test_freq=10",
        f"trainer.total_epochs={total_epochs}",
        *extra,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau2-root", type=Path, default=os.environ.get("TAU2_ROOT"))
    parser.add_argument("--verl-root", type=Path, default=os.environ.get("VERL_ROOT"))
    parser.add_argument(
        "--stage",
        choices=("smoke", "internal_dev", "full_train"),
        default="internal_dev",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra", action="append", default=[])
    args = parser.parse_args()

    if args.tau2_root is None or args.verl_root is None:
        raise RuntimeError("set TAU2_ROOT and VERL_ROOT, or pass both root arguments")
    project_root = Path(__file__).resolve().parents[1]
    _require_exact_checkout(args.tau2_root, TAU2_COMMIT, "Tau2")
    _require_exact_checkout(args.verl_root, VERL_COMMIT, "veRL")
    model_path = _require_env("MERGED_SFT_MODEL")
    _require_env("DEEPSEEK_USER_MODEL")
    _require_env("DEEPSEEK_JUDGE_MODEL")
    _require_env("DEEPSEEK_API_KEY")
    _require_env("DEEPSEEK_BASE_URL")

    parquet = project_root / "data" / "parquet"
    train_name = {
        "smoke": "airline_smoke.parquet",
        "internal_dev": "airline_rl_train.parquet",
        "full_train": "airline_official_train.parquet",
    }[args.stage]
    train_file = parquet / train_name
    val_file = parquet / "airline_internal_dev.parquet"
    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError("run scripts/prepare_tau2_dataset.py first")

    config_path = project_root / "configs" / "rl" / "airline_grpo_v1.yaml"
    os.environ["AGENTIC_RL_CONFIG"] = str(config_path)
    pythonpath = [
        str(project_root / "src"),
        str(args.tau2_root / "src"),
        str(args.verl_root),
    ]
    old_pythonpath = os.environ.get("PYTHONPATH")
    if old_pythonpath:
        pythonpath.append(old_pythonpath)
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)

    command = build_command(
        project_root=project_root,
        model_path=model_path,
        train_file=train_file,
        val_file=val_file,
        total_epochs=(1 if args.stage == "smoke" else args.epochs),
        extra=args.extra,
    )
    print(shlex.join(command))
    if not args.dry_run:
        subprocess.run(command, check=True, cwd=args.verl_root)


if __name__ == "__main__":
    main()

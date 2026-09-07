"""Validated launcher for the pinned veRL Tau2 Airline GRPO run."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from tau2_agentic_rl.base_identity import (
    capture_base_identity,
    find_base_identity,
    save_base_identity,
)
from tau2_agentic_rl.checkpoints import resolve_resume_path, restore_step_clock
from tau2_agentic_rl.config import expand_env, load_yaml
from tau2_agentic_rl.rl_resume import (
    MANIFEST,
    build_resume_identity,
    require_same_identity,
    save_resume_identity,
)
from tau2_agentic_rl.training_config import effective_project_config, training_overrides

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
    run_name: str = "qwen3_4b_airline_grpo_v1",
    run_root: Path | None = None,
    resume_from_path: Path | None = None,
    seed: int = 42,
    project_config: dict | None = None,
) -> list[str]:
    """Translate the plan to veRL v0.9.0 Hydra overrides."""
    if total_epochs < 1 or seed < 0:
        raise ValueError("epochs must be positive and seed must be nonnegative")
    agent_config = project_root / "configs" / "rl" / "agent_loop_v1.yaml"
    run_root = run_root or project_root / "outputs" / "runs" / run_name
    project = effective_project_config(
        project_config
        or load_yaml(
            Path(__file__).resolve().parents[1] / "configs/rl/airline_grpo_v1.yaml"
        ),
        extra,
    )
    mapped = training_overrides(project)
    if resume_from_path is not None:
        resume_from_path = resolve_resume_path(project_root, resume_from_path)
    resume_overrides = (
        [
            "trainer.resume_mode=resume_path",
            f"trainer.resume_from_path={resume_from_path}",
        ]
        if resume_from_path is not None
        else ["trainer.resume_mode=disable"]
    )
    command = [
        sys.executable,
        "-m",
        "tau2_agentic_rl.verl_entrypoint",
        "algorithm.rollout_correction.bypass_mode=true",
        "algorithm.rollout_correction.loss_type=ppo_clip",
        "algorithm.filter_groups.max_inflight_gen_batches=1",
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        "data.return_raw_chat=true",
        "data.filter_overlong_prompts=true",
        "data.truncation=error",
        "data.continuous_token.enable=false",
        f"data.seed={seed}",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.use_remove_padding=true",
        # veRL v0.9.0 interprets this field in prompt groups. Four prompts
        # times rollout.n=8 gives the planned 32-trajectory mini-batch.
        "actor_rollout_ref.actor.use_dynamic_bsz=true",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768",
        f"actor_rollout_ref.actor.data_loader_seed={seed}",
        "actor_rollout_ref.actor.fsdp_config.param_offload=false",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.calculate_log_probs=true",
        "actor_rollout_ref.rollout.load_format=safetensors",
        "actor_rollout_ref.rollout.layered_summon=true",
        f"actor_rollout_ref.rollout.seed={seed}",
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
        f"trainer.experiment_name={run_name}",
        f"trainer.default_local_dir={run_root / 'checkpoints'}",
        f"trainer.rollout_data_dir={run_root / 'verl_rollouts'}",
        "+trainer.max_attempts_without_update=50",
        *resume_overrides,
        "trainer.save_freq=10",
        "trainer.test_freq=10",
        f"trainer.total_epochs={total_epochs}",
        *[f"{key}={value}" for key, value in mapped.items()],
        *[item for item in extra if item.split("=", 1)[0].lstrip("+") not in mapped],
    ]
    return command


def _safe_run_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not normalized:
        raise ValueError("run name is empty after normalization")
    return normalized


def validate_run_destination(run_root: Path, resume_from_path: Path | None) -> None:
    """Check the ORIGINAL destination, without creating or changing any files."""
    if not run_root.exists() or not any(run_root.iterdir()):
        return
    if resume_from_path is None:
        raise FileExistsError(
            f"run directory is not empty: {run_root}; choose --run-name or "
            "explicitly pass --resume-from-path"
        )
    if resume_from_path.parent.parent.resolve() != run_root.resolve():
        raise ValueError("cannot resume another run into a nonempty run directory")


def prepare_run_directory(
    run_root: Path,
    resume_from_path: Path | None,
    model_path: str,
    project: dict,
    *,
    resume_identity: dict,
) -> Path:
    """Validate destination, base and runtime config before the first write."""
    import yaml

    validate_run_destination(run_root, resume_from_path)
    if resume_from_path is not None:
        # Check the source checkpoint even when the destination is a NEW run.
        require_same_identity(resume_from_path, resume_identity)
    if (run_root / MANIFEST).exists():
        require_same_identity(run_root, resume_identity)
    base_identity = capture_base_identity(model_path)
    if resume_from_path is not None:
        saved = json.loads(
            find_base_identity(resume_from_path).read_text(encoding="utf-8")
        )
        if saved["files"] != base_identity["files"]:
            raise ValueError("RL resume base identity changed")
    runtime_path = run_root / "runtime_config.yaml"
    if runtime_path.exists() and load_yaml(runtime_path) != project:
        raise ValueError("resume runtime config changed; choose a new --run-name")

    run_root.mkdir(parents=True, exist_ok=True)
    save_base_identity(run_root, base_identity)
    save_resume_identity(run_root, resume_identity)
    if not runtime_path.exists():
        runtime_path.write_text(
            yaml.safe_dump(project, sort_keys=False), encoding="utf-8"
        )
    return runtime_path


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name")
    parser.add_argument("--resume-from-path", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra", action="append", default=[])
    parser.add_argument("--config", type=Path)
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
    config_path = (args.config or config_path).resolve()
    project = effective_project_config(load_yaml(config_path), args.extra)
    if args.resume_from_path is not None:
        args.resume_from_path = resolve_resume_path(project_root, args.resume_from_path)
        restore_step_clock(
            args.resume_from_path,
            int(args.resume_from_path.name.split("global_step_")[1]),
        )
    generated_run_name = f"{args.stage}_lr{project['optimizer']['lr']}_seed{args.seed}"
    run_name = _safe_run_name(args.run_name or generated_run_name)
    run_root = project_root / "outputs" / "runs" / run_name
    if not args.dry_run:
        validate_run_destination(run_root, args.resume_from_path)
    output_env = {
        "TRAJECTORY_OUTPUT_DIR": run_root / "trajectories",
        "CHECKPOINT_OUTPUT_DIR": run_root / "checkpoints",
        "JUDGE_CACHE_DIR": run_root / "judge_cache",
        "USER_CACHE_DIR": run_root / "user_cache",
        "METRICS_OUTPUT_DIR": run_root / "metrics",
        "REPORTS_OUTPUT_DIR": run_root / "reports",
    }
    for name, path in output_env.items():
        os.environ[name] = str(path)
    os.environ["AGENTIC_RL_CONFIG"] = str(config_path)
    os.environ["AGENTIC_RL_PROJECT_ROOT"] = str(project_root)
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
        run_name=run_name,
        run_root=run_root,
        resume_from_path=args.resume_from_path,
        seed=args.seed,
        project_config=project,
    )
    print(shlex.join(command))
    if not args.dry_run:
        project = expand_env(project)
        resume_identity = build_resume_identity(
            project_root, project, command, stage=args.stage
        )
        runtime_path = prepare_run_directory(
            run_root,
            args.resume_from_path,
            model_path,
            project,
            resume_identity=resume_identity,
        )
        os.environ["AGENTIC_RL_CONFIG"] = str(runtime_path)
        launches = run_root / "launches"
        launches.mkdir(exist_ok=True)
        (launches / f"{uuid4().hex}.json").write_text(
            json.dumps({"command": command, "config": project}, indent=2),
            encoding="utf-8",
        )
        subprocess.run(command, check=True, cwd=args.verl_root)


if __name__ == "__main__":
    main()

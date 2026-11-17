"""Run the train30 12-update experiment with official tests at steps 6 and 12."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.export_verl_lora import adapter_dir
from scripts.summarize_evaluation import summarize
from tau2_agentic_rl.checkpoints import resolve_resume_path, restore_step_clock
from tau2_agentic_rl.config import current_profile, load_yaml
from tau2_agentic_rl.evaluation import evaluation_lock, fingerprint_directory
from tau2_agentic_rl.reward.truncation import ZERO_TRUNCATION
from tau2_agentic_rl.rl_resume import read_resume_identity, require_same_identity
from tau2_agentic_rl.user_simulation import filter_enabled
from tau2_agentic_rl.versions import sha256_json

PROJECT = Path(__file__).resolve().parents[1]
SEED = 42


def run_script(project: Path, script: str, args: list[str], *, pause_at=0) -> None:
    env = os.environ.copy()
    env.pop("TAU2_RL_PAUSE_AT_STEP", None)
    if pause_at:
        env["TAU2_RL_PAUSE_AT_STEP"] = str(pause_at)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(project), str(project / "src"), env.get("PYTHONPATH")) if p
    )
    subprocess.run(
        [sys.executable, "-u", str(project / "scripts" / script), *args],
        check=True, cwd=project, env=env,
    )


def verified_checkpoint(project: Path, run: Path, step: int) -> Path:
    checkpoint = resolve_resume_path(project, run / "checkpoints" / f"global_step_{step}")
    restore_step_clock(checkpoint, step)
    require_same_identity(checkpoint, read_resume_identity(run))
    return checkpoint


def require_checkpoint_space(run: Path, last_step: int, evaluations: Path, name: str) -> None:
    model = Path(os.environ["MERGED_SFT_MODEL"])
    # Both scheduled checkpoints contain the full base model. Export needs one
    # more full copy temporarily, even though only the adapter is retained.
    base_size = sum(path.stat().st_size for path in model.glob("*.safetensors"))
    if not base_size:
        raise FileNotFoundError(f"no merged model weights in {model}")
    pending_saves = sum(step > last_step for step in (6, 12))
    needs_export = any(
        not (evaluations / f"{name}_step{step}").exists()
        and not (run / f"official_eval_step{step}" / "lora_adapter").exists()
        for step in (6, 12)
    )
    minimum = base_size * (pending_saves + needs_export)
    target = next(parent for parent in (run, *run.parents) if parent.is_dir())
    free = shutil.disk_usage(target).free
    if free < minimum:
        gib = 1024 ** 3
        raise OSError(
            f"{target} has {free / gib:.1f} GiB free; "
            f"checkpoints and export alone need at least {minimum / gib:.1f} GiB"
        )


def run_experiment(project: Path, name: str, *, dry_run: bool, resume: bool) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("invalid experiment run name")
    if filter_enabled(load_yaml(current_profile(project, "official_evaluation"))):
        raise ValueError("official evaluation requires the user-simulator filter to be off")
    run = project / "outputs" / "runs" / name
    evaluations = project / "outputs" / "evaluations"
    train_args = [
        "--stage", "full_train", "--reward-judge", "--run-name", name, "--seed", str(SEED),
        "--extra", "trainer.total_training_steps=12",
        "--extra", "trainer.save_freq=6",
        "--extra", "trainer.test_freq=-1",
        # A recovery checkpoint between 6 and 12 must not evict step 6.
        "--extra", "trainer.max_actor_ckpt_to_keep=null",
    ]
    eval_args = [
        "--split", "official_test", "--samples", "4", "--no-reward-judge",
        "--max-refill-rounds", "0", "--seed", str(SEED),
    ]
    if dry_run:
        run_script(project, "train_airline_grpo.py", [*train_args, "--dry-run"], pause_at=6)
        run_script(project, "evaluate_airline.py", [
            *eval_args, "--tag", f"{name}_step6", "--dry-run",
        ])
        print("Plan: train to 6, official test (80), resume to 12, official test (80).", flush=True)
        return

    def evaluate_step(step: int) -> None:
        tag = f"{name}_step{step}"
        output = evaluations / tag
        export = run / f"official_eval_step{step}"
        adapter = export / "lora_adapter"
        print(f"Official evaluation at optimizer step {step}", flush=True)
        if not output.exists():
            if not adapter.exists():
                export.mkdir(parents=True, exist_ok=True)
                with TemporaryDirectory(prefix="merge-", dir=export) as directory:
                    target = Path(directory)
                    run_script(project, "export_verl_lora.py", [
                        "--stage", "rl", "--verl-root", os.environ["VERL_ROOT"],
                        "--local-dir", str(verified_checkpoint(project, run, step) / "actor"),
                        "--target-dir", str(target),
                    ])
                    adapter_dir(target).rename(adapter)
            run_script(project, "evaluate_airline.py", [
                *eval_args, "--tag", tag, "--lora-adapter", str(adapter_dir(export)),
            ])
        manifest = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
        identity = manifest["identity"]
        if (
            identity["split"] != "official_test" or identity["seed"] != SEED
            or identity["adapter_path"] != str(adapter.resolve())
            or identity["adapter_sha256"] != sha256_json(fingerprint_directory(adapter))
            or identity["truncation_policy"] != ZERO_TRUNCATION
            or identity["reward_judge_enabled"] or identity["user_sim_filter_enabled"]
        ):
            raise ValueError(f"unexpected official evaluation identity for {tag}")
        summary = summarize(output / "trajectories")
        if summary["valid_samples"] != 80 or summary["infrastructure_failures"]:
            raise ValueError(f"{tag} did not complete in exactly 80 attempts")
        summary_path = output / "summary.json"
        if not summary_path.exists():
            temporary_path = output / "summary.json.tmp"
            temporary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            temporary_path.replace(summary_path)

    control = project / "outputs" / "experiments" / name
    control.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(control):
        completed = sorted(
            int(path.name.removeprefix("global_step_"))
            for path in (run / "checkpoints").glob("global_step_*")
            if re.fullmatch(r"global_step_\d+", path.name)
        )
        if resume:
            if not completed or completed[-1] > 12:
                raise ValueError("resume requires a checkpoint from this 12-step experiment")
            step = completed[-1]
            checkpoint = verified_checkpoint(project, run, step)
        else:
            if run.exists() or any((evaluations / f"{name}_step{s}").exists() for s in (6, 12)):
                raise FileExistsError("use a new run name or --resume for this experiment")
            step, checkpoint = 0, None
        require_checkpoint_space(run, step, evaluations, name)
        if step < 6:
            continuation = ["--resume-from-path", str(checkpoint)] if checkpoint else []
            run_script(project, "train_airline_grpo.py", [*train_args, *continuation], pause_at=6)
        sixth = verified_checkpoint(project, run, 6)
        evaluate_step(6)
        if step < 12:
            run_script(project, "train_airline_grpo.py", [
                *train_args, "--resume-from-path", str(checkpoint if step >= 6 else sixth),
            ])
        verified_checkpoint(project, run, 12)
        evaluate_step(12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_experiment(PROJECT, args.run_name, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()

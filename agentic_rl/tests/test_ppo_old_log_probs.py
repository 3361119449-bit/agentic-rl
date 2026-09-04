from pathlib import Path

from scripts.train_airline_grpo import build_command


def test_launcher_uses_vllm_rollout_log_probs_as_fixed_old_policy() -> None:
    project_root = Path("C:/agentic-rl-test")
    command = build_command(
        project_root=project_root,
        model_path="model",
        train_file=project_root / "train.parquet",
        val_file=project_root / "dev.parquet",
        total_epochs=1,
        extra=[],
    )
    assert "actor_rollout_ref.rollout.calculate_log_probs=true" in command
    assert "algorithm.rollout_correction.bypass_mode=true" in command
    assert "actor_rollout_ref.actor.ppo_epochs=2" in command
    assert "actor_rollout_ref.actor.use_kl_loss=false" in command
    assert command[2] == "tau2_agentic_rl.verl_entrypoint"

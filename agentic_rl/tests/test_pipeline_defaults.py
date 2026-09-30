from pathlib import Path

import pytest

from scripts import evaluate_airline
from scripts.train_airline_grpo import build_command
from tau2_agentic_rl.config import load_yaml
from tau2_agentic_rl.training_config import effective_project_config

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "name", ["evaluation/airline_eval_v1.yaml", "rl/airline_grpo_v1.yaml"]
)
def test_actual_evaluation_configs_generate_explicit_bf16(name):
    args = evaluate_airline.parse_args(["--model-path", "/model"])
    project = load_yaml(ROOT / "configs" / name)
    command = evaluate_airline.build_evaluation_command(
        args,
        project_root=ROOT,
        data_file=Path("/data"),
        run_root=Path("/run"),
        project=project,
        identity={"temperature": 1, "top_p": 1, "top_k": -1},
    )
    assert "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16" in command
    assert "actor_rollout_ref.actor.fsdp_config.dtype=bfloat16" in command
    assert "actor_rollout_ref.rollout.dtype=bfloat16" in command


def test_default_training_command_uses_v4():
    command = build_command(
        project_root=ROOT,
        model_path="/model",
        train_file=Path("/train"),
        val_file=Path("/val"),
        total_epochs=1,
        extra=[],
    )
    assert "+algorithm.procredit_enabled=true" in command
    assert "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16" in command


def test_evaluation_defaults_to_strict_and_has_explicit_official_only_switch():
    assert evaluate_airline.parse_args([]).reward_judge is True
    assert evaluate_airline.parse_args(["--no-reward-judge"]).reward_judge is False


@pytest.mark.parametrize(
    "override",
    [
        "trainer.critic_warmup=100",
        "+trainer.critic_warmup=1",
        "~trainer.critic_warmup=null",
        "trainer={critic_warmup:100}",
    ],
)
def test_critic_warmup_cannot_skip_actor_and_advance_optimizer_counter(override):
    with pytest.raises(ValueError, match="critic_warmup"):
        effective_project_config(
            load_yaml(ROOT / "configs/rl/airline_procredit_v4.yaml"), [override]
        )


def test_zero_critic_warmup_remains_supported():
    effective_project_config(
        load_yaml(ROOT / "configs/rl/airline_procredit_v4.yaml"),
        ["trainer.critic_warmup=0"],
    )

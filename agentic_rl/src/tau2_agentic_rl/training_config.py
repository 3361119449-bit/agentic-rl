"""Single mapping from project YAML to pinned veRL Hydra training options."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

TRAINING_KEYS = {
    "algorithm.adv_estimator": "algorithm.adv_estimator",
    "algorithm.norm_adv_by_std_in_grpo": "algorithm.norm_adv_by_std_in_grpo",
    "algorithm.use_kl_in_reward": "algorithm.use_kl_in_reward",
    "algorithm.kl_coef": "algorithm.kl_ctrl.kl_coef",
    "algorithm.use_kl_loss": "actor_rollout_ref.actor.use_kl_loss",
    "algorithm.clip_ratio_low": "actor_rollout_ref.actor.clip_ratio_low",
    "algorithm.clip_ratio_high": "actor_rollout_ref.actor.clip_ratio_high",
    "algorithm.clip_ratio_c": "actor_rollout_ref.actor.clip_ratio_c",
    "algorithm.loss_agg_mode": "actor_rollout_ref.actor.loss_agg_mode",
    "algorithm.entropy_coeff": "actor_rollout_ref.actor.entropy_coeff",
    "algorithm.grad_clip": "actor_rollout_ref.actor.grad_clip",
    "lora.rank": "actor_rollout_ref.model.lora_rank",
    "lora.alpha": "actor_rollout_ref.model.lora_alpha",
    "lora.target_modules": "actor_rollout_ref.model.target_modules",
    "hardware.gradient_checkpointing": "actor_rollout_ref.model.enable_gradient_checkpointing",
    "hardware.tensor_parallel_size": "actor_rollout_ref.rollout.tensor_model_parallel_size",
    "optimizer.lr": "actor_rollout_ref.actor.optim.lr",
    "ppo.ppo_epochs": "actor_rollout_ref.actor.ppo_epochs",
    "ppo.verl_ppo_mini_batch_size_prompts": "actor_rollout_ref.actor.ppo_mini_batch_size",
    "ppo.ppo_micro_batch_size_per_gpu": "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
    "ppo.log_prob_micro_batch_size_per_gpu": "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu",
    "rollout.max_context_length": "actor_rollout_ref.rollout.max_model_len",
    "rollout.temperature": "actor_rollout_ref.rollout.temperature",
    "rollout.top_p": "actor_rollout_ref.rollout.top_p",
    "rollout.top_k": "actor_rollout_ref.rollout.top_k",
    "rollout.group_size": "actor_rollout_ref.rollout.n",
    "rollout.agent_worker_count": "actor_rollout_ref.rollout.agent.num_workers",
    "rollout.vllm_max_num_seqs": "actor_rollout_ref.rollout.max_num_seqs",
    "dynamic_sampling.enable": "algorithm.filter_groups.enable",
    "dynamic_sampling.metric": "algorithm.filter_groups.metric",
    "dynamic_sampling.train_batch_size": "data.train_batch_size",
    "dynamic_sampling.max_num_gen_batches": "algorithm.filter_groups.max_num_gen_batches",
    "vllm.gpu_memory_utilization": "actor_rollout_ref.rollout.gpu_memory_utilization",
}


def hydra_value(value: Any) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


def training_overrides(project: dict[str, Any]) -> dict[str, str]:
    result = {}
    for source, target in TRAINING_KEYS.items():
        section, key = source.split(".")
        result[target] = hydra_value(project[section][key])
    result["data.max_response_length"] = str(project["rollout"]["max_context_length"])
    result["actor_rollout_ref.actor.kl_loss_coef"] = str(
        project["algorithm"]["kl_coef"]
    )
    return result


def effective_project_config(
    project: dict[str, Any], extra: list[str]
) -> dict[str, Any]:
    import yaml

    result = deepcopy(project)
    reverse = {target: source for source, target in TRAINING_KEYS.items()}
    for override in extra:
        key, separator, raw = override.partition("=")
        normalized = key.lstrip("+")
        if not separator:
            raise ValueError("--extra must use key=value")
        fixed = {
            "algorithm.rollout_correction.bypass_mode": True,
            "algorithm.rollout_correction.loss_type": "ppo_clip",
            "actor_rollout_ref.rollout.calculate_log_probs": True,
            "data.continuous_token.enable": False,
        }
        if normalized in fixed and yaml.safe_load(raw) != fixed[normalized]:
            raise ValueError(
                f"{key} conflicts with the pinned token/old-policy contract"
            )
        if normalized in {
            "data.max_response_length",
            "actor_rollout_ref.actor.kl_loss_coef",
        }:
            raise ValueError(
                f"{key} is derived from project YAML; edit its source there"
            )
        if normalized in {
            "trainer.resume_mode",
            "trainer.resume_from_path",
            "trainer.default_local_dir",
            "trainer.experiment_name",
            "trainer.rollout_data_dir",
            "actor_rollout_ref.model.path",
            "data.train_files",
            "data.val_files",
        }:
            raise ValueError(f"use dedicated CLI arguments instead of --extra {key}")
        if separator and normalized in reverse:
            section, field = reverse[normalized].split(".")
            result[section][field] = yaml.safe_load(raw)
    dynamic, rollout, ppo = result["dynamic_sampling"], result["rollout"], result["ppo"]
    if not dynamic["enable"] or dynamic["metric"] != "train_reward":
        raise ValueError(
            "pinned capped trainer requires dynamic sampling on train_reward"
        )
    if result["algorithm"]["adv_estimator"] != "grpo":
        raise ValueError("this trainer requires GRPO advantages")
    if ppo["ppo_epochs"] < 1 or result["optimizer"]["lr"] <= 0:
        raise ValueError("PPO epochs and learning rate must be positive")
    if (
        dynamic["train_batch_size"],
        dynamic["conceptual_gen_batch_size"],
        dynamic["max_num_gen_batches"],
        dynamic["group_size"],
        rollout["group_size"],
    ) != (4, 8, 3, 8, 8):
        raise ValueError(
            "the pinned capped trainer requires 4 groups, 8 trajectories, 3 batches of 8"
        )
    if dynamic["max_rollouts_per_optimizer_step"] != 192:
        raise ValueError("max_rollouts_per_optimizer_step must equal 3 * 8 * 8")
    if (
        ppo["effective_trajectory_mini_batch_size"]
        != ppo["verl_ppo_mini_batch_size_prompts"] * rollout["group_size"]
    ):
        raise ValueError("PPO prompt/trajectory minibatch sizes disagree")
    if result["lora"]["dropout"] != 0:
        raise ValueError("pinned rollout log-prob path requires LoRA dropout=0")
    if (
        result["algorithm"]["use_kl_in_reward"]
        or result["algorithm"]["use_kl_loss"]
        or result["algorithm"]["kl_coef"] != 0
    ):
        raise ValueError("this experiment implements PPO clipping with KL disabled")
    if ppo["old_log_probs_source"] != "vllm_rollout":
        raise ValueError("old log probabilities must come from vLLM rollout")
    return result

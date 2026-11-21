# Tau2 Airline 官方奖励 GRPO 120 步对照

本页归档 2026-11-17 至 2026-11-19 完成的训练。该实验从 SFT v4 step 334 的 BF16 合并模型开始，使用 Tau2 官方任务完成奖励。实际训练代码见 [部署提交](https://github.com/3361119449-bit/agentic-rl/commit/3343941551e59fc9ec1cde87ececa85615491bb2)，SFT 基座文件和训练数据选择以文末指纹为准。

## 训练阶段

| 实际优化器步 | 启动权重 | 轨迹并发 | vLLM 序列上限 | vLLM 显存预算 |
| --- | --- | ---: | ---: | ---: |
| 1–60 | SFT v4 step 334 | 8 | 8 | 0.40 |
| 61–120 | 第 60 步 checkpoint，恢复模型、优化器和数据加载状态 | 32 | 32 | 0.70 |

第二段以 `trainer.total_training_steps=180` 初始化训练器，控制器实际在 optimizer step 120 完成训练和最后一次测评后退出。每 20 步测评一次。续训阶段的中间 checkpoint 在下一段恢复后清理，测评 adapter 在结果核验后清理，最终保留第 120 步 checkpoint。

## 训练和测评口径

- 修复后的 30 个 Airline 训练任务，seed 42。每步首批 4 组，每组 8 条。最多三波组采样，过滤组内优势全零的组。补采到上限时可用已评分的有效成员组成更新批次。
- 使用 GRPO 组内归一化、PPO clipping 下侧 0.20 和上侧 0.28、token-mean 损失、1 个 PPO epoch。无 Actor KL loss 或 reward KL。PPO mini-batch 为 4 个任务组，动态训练微批预算 32,768 tokens。
- 单张 A800 80 GB，TP=1，rank 32、alpha 64、all-linear LoRA。基座、Actor 计算和 rollout 用 BF16，LoRA 主参数、归约、缓冲和 AdamW 状态用 FP32。AdamW 学习率 `5e-6`，weight decay 0.01，betas `(0.9, 0.999)`，epsilon `1e-8`，恒定学习率。
- 训练和本轮官方测评使用 24,576 token 总上下文、35 个助手轮次，每轮最多一个工具调用。训练截断时标记计零，每槽最多重采两次。测评截断计零，重采数为零。
- 训练和测评均关闭 Agent 奖励 Judge 与用户合规筛选。DeepSeek Flash 作为用户模拟器，温度 0。客服采样温度 1.0，`top_p=1.0`，`top_k=-1`。用户 API 并发为 8。
- 第 20、40、60、80、100、120 步均使用 20 个固定官方测试任务，每任务四条。测评轨迹并发为 8，vLLM 显存预算为 0.40。主指标按截断计零，原始官方分保留在汇总中。

现行训练入口由 [正式配置清单](../../configs/current.json) 选取。下方 JSON 来自本次服务器实际运行快照。第二阶段覆盖字段叠加在第一阶段设置上。主机路径用环境占位符表示，运行生成的输出目录由启动器创建。

## 运行配置快照

```json
{
  "actual_schedule": {
    "continuation_eval_adapters_retained": [],
    "continuation_run_checkpoint_retention": [
      120
    ],
    "evaluation_steps": [
      20,
      40,
      60,
      80,
      100,
      120
    ],
    "phase_2_training_argv_total_training_steps_is_180": true,
    "segments": [
      {
        "actual_last_optimizer_step": 60,
        "from_checkpoint_step": 0
      },
      {
        "actual_last_optimizer_step": 120,
        "checkpoint_controller_stop_after_evaluation": 120,
        "from_checkpoint_step": 60
      }
    ]
  },
  "archive_kind": "historical-run-config",
  "base_model_identity": {
    "chat_template_sha256": "caacf1d133668476b9af859b92932796d536fe3f8323e2a8fb4ac55a28541270",
    "files_sha256": {
      "chat_template.jinja": "64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326",
      "config.json": "74e923dd507a5ecec8d596353290ca705ef8e4b7191d5823bbd4b77040515012",
      "generation_config.json": "4d9818c3d27895c0058828a5f68bc7a4de80c3ae1bcdac90936ce24178063f59",
      "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
      "model-00001-of-00002.safetensors": "cc5f1128e140f2c774fd1a7f35a712329c3b330173746bea4e3b976811c741f1",
      "model-00002-of-00002.safetensors": "613e45039a435509370af97315361e3ad90fc04eee9244ddfc7b788de07c315f",
      "model.safetensors.index.json": "06b3d5319b6d76d1a4a2433419180016cfd54ed62d086a5e6567a809f8c82634",
      "special_tokens_map.json": "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd",
      "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
      "tokenizer_config.json": "4b5f2f80f84faefe8420e1616671adb1dd3d7e632038d34b1f0e3a1363a51059",
      "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
    },
    "name": "Qwen3-4B, SFT v4 step 334, merged BF16"
  },
  "official_evaluation_identity": {
    "physical_rollout_limit_per_evaluation": 80,
    "reward_judge_enabled": false,
    "samples_per_task": 4,
    "seed": 42,
    "split": "official_test",
    "task_ids": [
      "2",
      "6",
      "8",
      "13",
      "16",
      "18",
      "19",
      "22",
      "24",
      "25",
      "26",
      "29",
      "30",
      "31",
      "32",
      "35",
      "37",
      "44",
      "45",
      "48"
    ],
    "temperature": 1,
    "top_k": -1,
    "top_p": 1,
    "truncation_policy": "zero-trajectory-v1",
    "truncation_resamples": 0,
    "user_model": "deepseek/deepseek-flash",
    "user_sim_filter_enabled": false
  },
  "official_evaluation_runtime": {
    "agent_policy": {
      "path": "configs/prompts/airline_sft_v4_single_tool.v1.json",
      "system_prompt_sha256": "54c8a8153c0f9fe4d2bc4d57a3698221d09388d84245afdcc253378ee4238cb2"
    },
    "judge": {
      "enabled": false
    },
    "model": {
      "base": "Qwen/Qwen3-4B-Instruct-2507",
      "sft_input": "${HOST_LOCAL_PATH}"
    },
    "precision": {
      "buffer_dtype": "float32",
      "model_dtype": "bfloat16",
      "param_dtype": "bfloat16",
      "reduce_dtype": "float32",
      "rollout_dtype": "bfloat16"
    },
    "project": {
      "annotation_version": "v5-trajectory-semantic-dedup",
      "domain": "airline",
      "reward_version": "official-35turn-zero-truncation",
      "tau2_commit": "a2c024725189473d2d7cea3a5cfdbcc67478e41f",
      "verl_commit": "483b8a009ba3a97563edee3a19887e4862b8094a"
    },
    "reward": {
      "truncation_multiplier": 0,
      "truncation_policy": "zero-trajectory-v1"
    },
    "rollout": {
      "agent_worker_count": 8,
      "group_size": 4,
      "initial_prompt_max_tokens": 24576,
      "judge_api_max_inflight": 8,
      "max_active_trajectories": 8,
      "max_context_length": 24576,
      "max_hard_turns": 35,
      "max_soft_turns": 35,
      "max_tool_calls_per_turn": 1,
      "queue_stall_timeout_seconds": 1800,
      "queue_timeout_seconds": null,
      "temperature": 1,
      "tool_parser": "hermes",
      "top_k": -1,
      "top_p": 1,
      "user_api_max_inflight": 8,
      "vllm_max_num_seqs": 8
    },
    "user_sim_filter": {
      "enabled": false
    },
    "user_simulator": {
      "llm_args": {
        "api_base": "https://api.deepseek.com",
        "extra_body": {
          "thinking": {
            "type": "disabled"
          }
        },
        "timeout": 120
      },
      "max_retries": 2,
      "model": "deepseek/deepseek-flash",
      "provider": "DeepSeek",
      "temperature": 0,
      "thinking": "disabled"
    }
  },
  "seed": 42,
  "status": "completed",
  "training_cli_overrides_delta_steps_61_120": {
    "actor_rollout_ref.rollout.gpu_memory_utilization": "0.7",
    "actor_rollout_ref.rollout.max_num_seqs": "32",
    "trainer.total_epochs": "30",
    "trainer.total_training_steps": "180"
  },
  "training_cli_overrides_steps_1_60": {
    "actor_rollout_ref.actor.clip_ratio_c": "10.0",
    "actor_rollout_ref.actor.clip_ratio_high": "0.28",
    "actor_rollout_ref.actor.clip_ratio_low": "0.2",
    "actor_rollout_ref.actor.data_loader_seed": "42",
    "actor_rollout_ref.actor.entropy_coeff": "0.0",
    "actor_rollout_ref.actor.fsdp_config.dtype": "bfloat16",
    "actor_rollout_ref.actor.fsdp_config.mixed_precision": "{param_dtype:bfloat16,reduce_dtype:float32,buffer_dtype:float32}",
    "actor_rollout_ref.actor.fsdp_config.model_dtype": "bfloat16",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload": "false",
    "actor_rollout_ref.actor.fsdp_config.param_offload": "false",
    "actor_rollout_ref.actor.fsdp_config.wrap_policy.tau2_fp32_lora_master": "true",
    "actor_rollout_ref.actor.fsdp_config.wrap_policy.tau2_strict_microbatch_tokens": "32768",
    "actor_rollout_ref.actor.grad_clip": "1.0",
    "actor_rollout_ref.actor.kl_loss_coef": "0.0",
    "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
    "actor_rollout_ref.actor.optim.betas": "[0.9,0.999]",
    "actor_rollout_ref.actor.optim.clip_grad": "1.0",
    "actor_rollout_ref.actor.optim.lr": "5e-06",
    "actor_rollout_ref.actor.optim.lr_scheduler_type": "constant",
    "actor_rollout_ref.actor.optim.override_optimizer_config": "{eps:1e-08}",
    "actor_rollout_ref.actor.optim.weight_decay": "0.01",
    "actor_rollout_ref.actor.ppo_epochs": "1",
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": "32768",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": "1",
    "actor_rollout_ref.actor.ppo_mini_batch_size": "4",
    "actor_rollout_ref.actor.use_dynamic_bsz": "true",
    "actor_rollout_ref.actor.use_kl_loss": "false",
    "actor_rollout_ref.model.enable_gradient_checkpointing": "true",
    "actor_rollout_ref.model.lora_alpha": "64",
    "actor_rollout_ref.model.lora_rank": "32",
    "actor_rollout_ref.model.target_modules": "all-linear",
    "actor_rollout_ref.model.use_remove_padding": "true",
    "actor_rollout_ref.rollout.agent.default_agent_loop": "tau2_airline",
    "actor_rollout_ref.rollout.agent.num_workers": "8",
    "actor_rollout_ref.rollout.calculate_log_probs": "true",
    "actor_rollout_ref.rollout.dtype": "bfloat16",
    "actor_rollout_ref.rollout.gpu_memory_utilization": "0.4",
    "actor_rollout_ref.rollout.layered_summon": "false",
    "actor_rollout_ref.rollout.load_format": "safetensors",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": "1",
    "actor_rollout_ref.rollout.max_model_len": "24576",
    "actor_rollout_ref.rollout.max_num_seqs": "8",
    "actor_rollout_ref.rollout.mode": "async",
    "actor_rollout_ref.rollout.n": "8",
    "actor_rollout_ref.rollout.name": "vllm",
    "actor_rollout_ref.rollout.seed": "42",
    "actor_rollout_ref.rollout.temperature": "1.0",
    "actor_rollout_ref.rollout.tensor_model_parallel_size": "1",
    "actor_rollout_ref.rollout.top_k": "-1",
    "actor_rollout_ref.rollout.top_p": "1.0",
    "algorithm.adv_estimator": "grpo",
    "algorithm.filter_groups.enable": "true",
    "algorithm.filter_groups.max_inflight_gen_batches": "1",
    "algorithm.filter_groups.max_num_gen_batches": "3",
    "algorithm.filter_groups.metric": "train_reward",
    "algorithm.kl_ctrl.kl_coef": "0.0",
    "algorithm.norm_adv_by_std_in_grpo": "true",
    "algorithm.rollout_correction.bypass_mode": "true",
    "algorithm.rollout_correction.loss_type": "ppo_clip",
    "algorithm.use_kl_in_reward": "false",
    "data.continuous_token.enable": "false",
    "data.filter_overlong_prompts": "true",
    "data.max_prompt_length": "24576",
    "data.max_response_length": "24576",
    "data.return_raw_chat": "true",
    "data.seed": "42",
    "data.train_batch_size": "4",
    "data.truncation": "error",
    "trainer.critic_warmup": "0",
    "trainer.logger": "[console,wandb]",
    "trainer.max_actor_ckpt_to_keep": "null",
    "trainer.max_attempts_without_update": "50",
    "trainer.n_gpus_per_node": "1",
    "trainer.nnodes": "1",
    "trainer.project_name": "tau2_airline_agentic_rl",
    "trainer.save_freq": "20",
    "trainer.test_freq": "-1",
    "trainer.total_epochs": "15",
    "trainer.total_training_steps": "60",
    "trainer.use_v1": "true",
    "trainer.v1.trainer_mode": "sync",
    "trainer.val_before_train": "false"
  },
  "training_code_snapshot": {
    "deployed_git_commit": "3343941551e59fc9ec1cde87ececa85615491bb2",
    "source_tar_gz_sha256": "4acd6e90914ec2c78a824e714084bfea9e8d922afef81a285fa33e0b0d54bf14"
  },
  "training_runtime_section_overrides_steps_61_120": {
    "rollout": {
      "agent_worker_count": 8,
      "group_size": 8,
      "initial_prompt_max_tokens": 24576,
      "judge_api_max_inflight": 8,
      "max_active_trajectories": 32,
      "max_context_length": 24576,
      "max_hard_turns": 35,
      "max_soft_turns": 35,
      "max_tool_calls_per_turn": 1,
      "queue_stall_timeout_seconds": 1800,
      "queue_timeout_seconds": null,
      "temperature": 1,
      "tool_parser": "hermes",
      "top_k": -1,
      "top_p": 1,
      "user_api_max_inflight": 8,
      "vllm_max_num_seqs": 32
    },
    "vllm": {
      "gpu_memory_utilization": 0.7
    }
  },
  "training_runtime_steps_1_60": {
    "agent_policy": {
      "path": "configs/prompts/airline_sft_v4_single_tool.v1.json",
      "system_prompt_sha256": "54c8a8153c0f9fe4d2bc4d57a3698221d09388d84245afdcc253378ee4238cb2"
    },
    "algorithm": {
      "adv_estimator": "grpo",
      "clip_ratio_c": 10,
      "clip_ratio_high": 0.28,
      "clip_ratio_low": 0.2,
      "entropy_coeff": 0,
      "grad_clip": 1,
      "kl_coef": 0,
      "loss_agg_mode": "token-mean",
      "norm_adv_by_std_in_grpo": true,
      "use_kl_in_reward": false,
      "use_kl_loss": false
    },
    "annotations": {
      "action_dependencies": "data/annotations/airline_action_dependencies.train.v1.json",
      "policy_rules": "data/annotations/airline_mandatory_policy_rules.train.v1.json",
      "required_actions": "data/annotations/airline_required_actions.train.repaired.v1.json",
      "semantic_checks": "data/annotations/airline_semantic_checks.train.repaired.v1.json",
      "task_judge_guidance": "data/annotations/airline_task_judge_guidance.train.v1.json",
      "transfer_rules": "data/annotations/airline_transfer_rules.train.v1.json"
    },
    "dynamic_sampling": {
      "conceptual_gen_batch_size": 4,
      "criterion": "final_advantage_nonzero",
      "enable": true,
      "group_size": 8,
      "max_num_gen_batches": 3,
      "max_rollouts_per_optimizer_step": 96,
      "metric": "train_reward",
      "signal_tolerance": 1e-8,
      "train_batch_size": 4
    },
    "hardware": {
      "gpu": "1x_A800_80GB",
      "gradient_checkpointing": true,
      "tensor_parallel_size": 1
    },
    "judge": {
      "enabled": false
    },
    "lora": {
      "alpha": 64,
      "dropout": 0,
      "rank": 32,
      "target_modules": "all-linear"
    },
    "model": {
      "base": "Qwen/Qwen3-4B-Instruct-2507",
      "rl_method": "lora",
      "sft_input": "${HOST_LOCAL_PATH}"
    },
    "optimizer": {
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-8,
      "lr": 0.000005,
      "lr_scheduler_type": "constant",
      "weight_decay": 0.01
    },
    "ppo": {
      "effective_trajectory_mini_batch_size": 32,
      "log_prob_micro_batch_size_per_gpu": 1,
      "max_tokens_per_microbatch": 32768,
      "old_log_probs_source": "vllm_rollout",
      "ppo_epochs": 1,
      "ppo_micro_batch_size_per_gpu": 1,
      "verl_ppo_mini_batch_size_prompts": 4
    },
    "precision": {
      "buffer_dtype": "float32",
      "lora_master_dtype": "float32",
      "model_dtype": "bfloat16",
      "param_dtype": "bfloat16",
      "reduce_dtype": "float32",
      "rollout_dtype": "bfloat16"
    },
    "project": {
      "annotation_version": "v7-repaired-airline-train30",
      "domain": "airline",
      "reward_version": "official-repaired-train30-grpo-v1",
      "tau2_commit": "a2c024725189473d2d7cea3a5cfdbcc67478e41f",
      "verl_commit": "483b8a009ba3a97563edee3a19887e4862b8094a"
    },
    "reward": {
      "mode": "legacy",
      "progress_version": "progress-v3",
      "truncation_multiplier": 0,
      "truncation_policy": "zero-trajectory-v1"
    },
    "rollout": {
      "agent_worker_count": 8,
      "group_size": 8,
      "initial_prompt_max_tokens": 24576,
      "judge_api_max_inflight": 8,
      "max_active_trajectories": 8,
      "max_context_length": 24576,
      "max_hard_turns": 35,
      "max_soft_turns": 35,
      "max_tool_calls_per_turn": 1,
      "queue_stall_timeout_seconds": 1800,
      "queue_timeout_seconds": null,
      "temperature": 1,
      "tool_parser": "hermes",
      "top_k": -1,
      "top_p": 1,
      "user_api_max_inflight": 8,
      "vllm_max_num_seqs": 8
    },
    "slot_recovery": {
      "max_resamples": 2,
      "max_scoring_retries": 2,
      "max_truncation_resamples": 2,
      "on_exhaustion": "salvage_group"
    },
    "training_selection": {
      "identity": {
        "manifest_sha256": "a380903a7fab49697da5fc60c79ae5b153b18295f4d145e656c92358adef0621",
        "selection_sha256": "90b94ff5cbeb213c422e4bf215817a1c759101bcadd99ffc9ad831649cd17eea"
      },
      "mode": "task_or_local_credit",
      "split_file": "data/splits/airline_repaired_train30.v1.json",
      "task_repairs": "task_repairs/tasks.json"
    },
    "training_variant": "train30-official-grpo-v1",
    "user_sim_filter": {
      "base_url": "https://api.deepseek.com",
      "enabled": false,
      "max_resamples": 2,
      "max_retries": 2,
      "model": "deepseek-flash",
      "provider": "DeepSeek",
      "version": "user-sim-compliance-v1"
    },
    "user_simulator": {
      "llm_args": {
        "api_base": "https://api.deepseek.com",
        "extra_body": {
          "thinking": {
            "type": "disabled"
          }
        },
        "timeout": 120
      },
      "max_retries": 2,
      "model": "deepseek/deepseek-flash",
      "provider": "DeepSeek",
      "temperature": 0,
      "thinking": "disabled"
    },
    "vllm": {
      "gpu_memory_utilization": 0.4
    }
  }
}
```

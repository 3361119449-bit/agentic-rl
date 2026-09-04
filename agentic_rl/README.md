# Tau2 Airline Agentic RL

这是《Tau2 Airline Agentic RL 训练计划（完善版）》的可执行实现。项目使用固定版 Tau2 Airline 环境、Qwen3-4B SFT 合并模型和新的 RL LoRA，在 veRL v0.9.0 上进行 GRPO / DAPO-style 训练。

三套数值始终分开保存：

- `official_scores.reward`：Tau2 官方奖励；
- `custom_reward.strict_success`：本项目严格成功；
- `custom_reward.train_reward`：只用于 GRPO 的训练奖励。

`R_tau2_official != S_custom_strict != R_train`，最终报告脚本只输出 pass@1 和 pass@4，不把训练奖励冒充官方成功率。

## 已实现的关键约束

- Tau2 固定提交：`a2c024725189473d2d7cea3a5cfdbcc67478e41f`；
- veRL 固定提交：`483b8a009ba3a97563edee3a19887e4862b8094a`（v0.9.0）；
- 每条 rollout 创建独立 `AgentGymEnv` 和 Airline DB；
- Qwen 原始生成 token ID、vLLM old log-prob 和 turn 边界原样保存；
- 只有 Qwen 输出 token 的 `response_mask=1`，工具、用户、模板 token 均为 0；
- 16,384 token 生成前预算检查，15 轮软阈值、24 轮硬终止；
- DeepSeek 用户模拟器与 Judge 使用独立 prompt、缓存和重试；
- Judge 每条完整轨迹只调用一次，输出二值条目，并严格核对 criterion ID；
- 官方 DB/COMMUNICATE、必须动作、语义、强制 Policy gate 和过程扣分的双分支奖励；
- GRPO 组大小 8、组内标准差归一化、Clip-Higher 0.20/0.28、dual clip 10；
- PPO epoch=2、old policy 固定为 vLLM rollout log-prob、token-mean、无 KL、无 Critic；
- RL LoRA `r=32, alpha=64, dropout=0, all-linear`；
- 自定义有界动态采样：每个候选批 8 个任务组，每组 8 条轨迹，最多 3 批（192 条）；若不足 4 个混合奖励组，该候选优化步不更新参数，清空后换一批任务；
- train/internal-dev/test 物理分离，test 标注与训练配置分开；
- 每条轨迹原子化保存，可离线重新打分。

## 目录

```text
configs/                 RL、AgentLoop、冻结测试配置
data/annotations/        train/test 分开的动作、语义、Policy、转接标注
data/splits/             固定 24 train + 6 internal dev + 20 test
src/tau2_agentic_rl/     AgentLoop、Tau2 适配、奖励、Judge、动态采样
scripts/                 数据准备、训练、评估、审计、LoRA 合并
tests/                   CPU 单元测试
```

## AutoDL 安装

建议把本项目放在 Tau2 仓库的 `agentic_rl/` 目录；下面假定 Tau2 仓库为 `/root/tau2-bench-data`。

```bash
export TAU2_ROOT=/root/tau2-bench-data
export AGENTIC_RL_ROOT=$TAU2_ROOT/agentic_rl
export VERL_ROOT=/root/verl

git -C "$TAU2_ROOT" checkout a2c024725189473d2d7cea3a5cfdbcc67478e41f
git clone https://github.com/verl-project/verl.git "$VERL_ROOT"
git -C "$VERL_ROOT" checkout 483b8a009ba3a97563edee3a19887e4862b8094a

python -m pip install -e "$TAU2_ROOT[gym]"
python -m pip install -e "$VERL_ROOT"
python -m pip install -e "$AGENTIC_RL_ROOT[data,test]"
```

请优先使用与 veRL v0.9.0、vLLM 和本机 CUDA 匹配的 AutoDL 镜像。正式运行前通过 `capture_versions.py` 固化实际软件版本。

## 环境变量

复制示例后填值：

```bash
cd "$AGENTIC_RL_ROOT"
cp .env.example .env
set -a
source .env
set +a
```

必须手工填写：

- `MERGED_SFT_MODEL`：AReaL Airline SFT LoRA 合并后的完整模型目录；
- `DEEPSEEK_USER_MODEL`：服务商中 DeepSeek Pro 的精确模型 ID；
- `DEEPSEEK_JUDGE_MODEL`：Judge 的精确模型 ID；
- `DEEPSEEK_BASE_URL` 与 `DEEPSEEK_API_KEY`。

代码会拒绝 `FIX_EXACT_MODEL_ID`。计划书没有给出精确 DeepSeek 型号，所以实现没有擅自猜测别名。不要提交 `.env`。

## 一次性数据与版本检查

```bash
cd "$AGENTIC_RL_ROOT"
python scripts/verify_setup.py \
  --tau2-data "$TAU2_ROOT/data/tau2/domains/airline"

python scripts/prepare_tau2_dataset.py

python scripts/capture_versions.py \
  --tau2-root "$TAU2_ROOT" \
  --verl-root "$VERL_ROOT" \
  --project-config configs/rl/airline_grpo_v1.yaml \
  --model-path "$MERGED_SFT_MODEL" \
  --output outputs/metadata/versions.json
```

固定划分为：官方 train 30 条，其中 RL train 24 条、internal dev 6 条；官方 test 20 条。`prepare_tau2_dataset.py` 生成 veRL 所需的 Parquet，生成物不提交 Git。

## SFT LoRA 合并

如果手里仍是 SFT adapter，先合并；RL 阶段会在合并模型上新挂一份 LoRA：

```bash
python scripts/merge_sft_lora.py \
  --sft-adapter /root/models/airline_sft_adapter \
  --output /root/models/qwen3_4b_airline_sft_merged
export MERGED_SFT_MODEL=/root/models/qwen3_4b_airline_sft_merged
```

## 分阶段运行

Stage -1，SFT 基线画像（每个官方 train task 8 条）：

```bash
python scripts/profile_sft_baseline.py \
  --tau2-root "$TAU2_ROOT" --verl-root "$VERL_ROOT"
```

完成后会把组内方差、全 0/全 1、组件均值、工具错误、长度和逐任务统计写入 `outputs/reports/sft_baseline_profile.json`。

Stage 0/1，先运行 CPU 不变量测试，再做真实 API/GPU 轨迹和人工 reward 审计：

```bash
python -m pytest
python scripts/run_reward_audit.py data/audits/reward_audit.v1.json
```

Stage 2，小规模 smoke：

```bash
python scripts/train_airline_grpo.py \
  --stage smoke --tau2-root "$TAU2_ROOT" --verl-root "$VERL_ROOT"
```

先用默认 `lr=5e-6`，只在单独 run 中比较 `1e-5`：

```bash
python scripts/train_airline_grpo.py \
  --stage smoke --tau2-root "$TAU2_ROOT" --verl-root "$VERL_ROOT" \
  --extra actor_rollout_ref.actor.optim.lr=1e-5
```

Stage 3，24 条任务训练并只看 6 条 internal dev：

```bash
python scripts/train_airline_grpo.py \
  --stage internal_dev --epochs 15 \
  --tau2-root "$TAU2_ROOT" --verl-root "$VERL_ROOT"
```

Stage 4，冻结配置后使用全部官方 train：

```bash
python scripts/train_airline_grpo.py \
  --stage full_train --epochs 15 \
  --tau2-root "$TAU2_ROOT" --verl-root "$VERL_ROOT"
```

Stage 5，只在最终冻结后运行 test；不做动态采样或参数更新：

```bash
python scripts/evaluate_airline.py \
  --split official_test --samples 4 --tag sft_baseline \
  --model-path "$MERGED_SFT_MODEL" \
  --tau2-root "$TAU2_ROOT" --verl-root "$VERL_ROOT"

python scripts/evaluate_airline.py \
  --split official_test --samples 4 --tag sft_grpo \
  --model-path "$MERGED_SFT_MODEL" \
  --lora-adapter outputs/checkpoints/FINAL_ADAPTER \
  --tau2-root "$TAU2_ROOT" --verl-root "$VERL_ROOT"
```

只汇总 pass@1 和 pass@4：

```bash
python scripts/summarize_evaluation.py \
  outputs/evaluations/sft_baseline/trajectories \
  --output outputs/reports/sft_baseline_pass1_pass4.json

python scripts/summarize_evaluation.py \
  outputs/evaluations/sft_grpo/trajectories \
  --output outputs/reports/sft_grpo_pass1_pass4.json
```

## 动态采样实现说明

veRL v0.9.0 的内置 V1 ReplayBuffer 会忽略 `algorithm.filter_groups.max_num_gen_batches`。本项目没有依赖该无上限行为，而是通过 `tau2_agentic_rl.verl_entrypoint` 构造 `CappedPPOTrainerSync`：

1. 同一 policy version 先生成 8 个 prompt group；
2. 每组由 8 条 trajectory 构成；
3. 仅保留 `train_reward` 组内不全同的 group；
4. 不足 4 组时再生成一批 8 组；
5. 最多 3 批，即 24 group / 192 trajectory；
6. 仍不足时清掉该候选批，不执行 backward/optimizer step，重新取任务；
7. 收满 4 组后用 32 条 trajectory 做两个 PPO epoch，再同步 vLLM 权重。

这段扩展依赖 veRL v0.9.0 的受保护接口，因此训练启动器会先检查完整 commit SHA；版本不符会直接终止。

## 离线重新打分

当只调整 reward 权重、过程扣分或必须动作标注，而且 Judge rubric 未变时，不必重新请求 DeepSeek：

```bash
python scripts/rescore_saved_trajectories.py \
  outputs/trajectories outputs/trajectories_rescored \
  --config configs/rl/airline_grpo_v1.yaml \
  --reward-version v2
```

## 正式训练前仍必须人工完成

- 填入 DeepSeek Pro 的精确、可复现模型 ID；
- 在 A800 上完成一次端到端 Stage 0，核对 Qwen3 工具标签与 tokenizer/chat template；
- 用真实 Airline 工具结果统计重新确认 1,024 token observation 预留量；
- 人工审计至少 50–100 条代表性轨迹及 Judge 结论；
- smoke 中确认更新前 ratio 接近 1、两个 PPO epoch 的 old log-prob 不变、checkpoint 可恢复且无 OOM；
- 在看任何 test reward 前冻结 reward、标注、prompt、训练步数和评估参数。

本地 CPU 测试不能替代上述 API、GPU 和人工验收。

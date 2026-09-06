# Tau2 Airline Agentic RL

这是《Tau2 Airline Agentic RL 训练计划（完善版）》的可执行实现。项目使用固定版 Tau2 Airline 环境、Qwen3-4B SFT 合并模型和新的 RL LoRA，在 veRL v0.9.0 上进行 GRPO / DAPO-style 训练。

三套数值始终分开保存：

- `official_scores.reward`：Tau2 官方奖励；
- `custom_reward.strict_success`：本项目严格成功；
- `custom_reward.train_reward`：只用于 GRPO 的训练奖励。

`R_tau2_official != S_custom_strict != R_train`，最终报告脚本只输出 pass@1 和 pass@4，不把训练奖励冒充官方成功率。

最新复审修复、评估断点续跑、PPO 审计开关与尚未完成的 GPU 验收见
[REVIEW_FOLLOWUP.md](REVIEW_FOLLOWUP.md)。CPU 测试通过不代表 GPU 端到端验收已完成。

## 已实现的关键约束

- Tau2 固定提交：`a2c024725189473d2d7cea3a5cfdbcc67478e41f`；
- veRL 固定提交：`483b8a009ba3a97563edee3a19887e4862b8094a`（v0.9.0）；
- 每条 rollout 创建独立 `AgentGymEnv` 和 Airline DB；
- Qwen3-4B-Instruct 的 JSON `<tool_call>` 固定使用 veRL `hermes` 解析器；
- 未知工具、非法 JSON/Schema 和多工具调用在进入 Tau2 前被确定性阻断；
- 写操作确认绑定到成功发送给用户的纯文本精确提案，只能消费一次；条件式或修改参数的 Yes 不授权旧操作；
- Qwen 原始生成 token ID、vLLM old log-prob 和 turn 边界原样保存；
- 只有 Qwen 输出 token 的 `response_mask=1`，工具、用户、模板 token 均为 0；
- 多轮 observation 包含 Qwen `turn_separator`，完整计入上下文预算；
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
- 每个实验使用独立 run 目录且默认禁用自动续训；
- 基础设施失败单独落盘，不计作模型失败或 pass@k 样本。
- 冻结评估身份与样本 slot；最终 test 必须补齐 20 × 4 = 80 条有效轨迹才输出分数。

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

下面从一台新的 AutoDL 实例开始，分别保留本仓库、Tau2、SFT veRL
和 RL veRL，避免两个 veRL 版本相互覆盖：

```bash
git lfs install
git clone https://github.com/3361119449-bit/agentic-rl.git /root/agentic-rl
git -C /root/agentic-rl lfs pull

git clone https://github.com/sierra-research/tau2-bench.git /root/tau2-bench-data
git -C /root/tau2-bench-data checkout a2c024725189473d2d7cea3a5cfdbcc67478e41f

git clone https://github.com/verl-project/verl.git /root/verl-sft-v071
git -C /root/verl-sft-v071 checkout bec9ef74768dd201881cd4e54cd0385e87caae27

git clone https://github.com/verl-project/verl.git /root/verl-rl-v090
git -C /root/verl-rl-v090 checkout 483b8a009ba3a97563edee3a19887e4862b8094a

export AGENTIC_REPO_ROOT=/root/agentic-rl
export TAU2_ROOT=/root/tau2-bench-data
export AGENTIC_RL_ROOT=$AGENTIC_REPO_ROOT/agentic_rl
export SFT_VERL_ROOT=/root/verl-sft-v071
export VERL_ROOT=/root/verl-rl-v090

python -m venv --system-site-packages /root/venvs/airline-sft
source /root/venvs/airline-sft/bin/activate
python -m pip install -e "$SFT_VERL_ROOT"
python -m pip install "transformers==4.57.1" pyarrow "peft==0.17.1" "torchdata==0.11.0"
python -m pip install -e "$AGENTIC_RL_ROOT"
deactivate

python -m venv --system-site-packages /root/venvs/airline-rl
source /root/venvs/airline-rl/bin/activate
python -m pip install -e "$TAU2_ROOT[gym]"
python -m pip install -e "$VERL_ROOT"
python -m pip install "vllm==0.24.0" "TransferQueue==0.1.8"
python -m pip install -e "$AGENTIC_RL_ROOT[data,test]"
```

不要把 v0.7.1 和 v0.9.0 依次 editable-install 到同一个 Python 环境；后装的
版本会覆盖前一个。下面 RL 命令默认已激活 `airline-rl`，SFT 训练和 SFT
adapter 导出则使用 `airline-sft`。

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

SFT 脚本保存的是 veRL/FSDP 检查点，不能直接传给 PEFT。先用固定的
veRL v0.7.1 导出标准 adapter；SFT 的 `--local-dir` 指向实际的
`global_step_N` 目录：

```bash
/root/venvs/airline-sft/bin/python "$AGENTIC_RL_ROOT/scripts/export_verl_lora.py" \
  --stage sft --verl-root "$SFT_VERL_ROOT" \
  --local-dir "$AGENTIC_REPO_ROOT/training/qwen3_4b_sft/runs/RUN_NAME/checkpoints/global_step_N" \
  --target-dir /root/models/qwen3_4b_sft_export
```

然后把导出的 SFT LoRA 合并到基础模型；RL 阶段会在该完整模型上新挂
一份 LoRA：

```bash
/root/venvs/airline-sft/bin/python scripts/merge_sft_lora.py \
  --sft-adapter /root/models/qwen3_4b_sft_export/lora_adapter \
  --output /root/models/qwen3_4b_airline_sft_merged
/root/venvs/airline-sft/bin/python scripts/verify_adapter_equivalence.py \
  --adapter /root/models/qwen3_4b_sft_export/lora_adapter \
  --merged-model /root/models/qwen3_4b_airline_sft_merged
export MERGED_SFT_MODEL=/root/models/qwen3_4b_airline_sft_merged
```

最后一条命令在 GPU 上顺序加载 adapter 版和合并版模型，比较同一固定输入
的末位 logits；超出给定数值误差时直接失败。

训练保存 `base_model_identity.json`（实际权重、tokenizer 文件、模板 SHA-256），
导出时复制到 adapter 目录。合并默认读取训练时的本地快照，不再解析 Hub main。
若快照搬家，可传 `--base-model /new/local/snapshot`，但文件内容必须一致。
旧 checkpoint 若缺少原始身份文件会拒绝导出；只能提供可追溯的原训练
`--base-identity /path/base_model_identity.json`，不能用今天的 Hub 模型冒充历史底座。

## 真实依赖契约与并发设置

在已安装固定 veRL/Ray/vLLM 的 RL 环境中，先执行（不加载权重、不请求 API）：

```bash
cd "$AGENTIC_RL_ROOT"
python -m pytest contract_tests -v
```

测试包含真实 runner 导入、训练/两种评估命令的 Hydra 组合及 `validate_config()`、
两个真实 Ray worker 的共享并发额度。GitHub Actions 单独运行此层；CPU 单元测试
不能替代它，也不能替代 GPU/API smoke。

两份 YAML 的 `rollout` 均区分 `agent_worker_count`、`max_active_trajectories`、
`user_api_max_inflight`、`judge_api_max_inflight` 和 `vllm_max_num_seqs`，默认均为 2。
前者只是 worker 数；轨迹在创建环境前领取同一个 Ray job 的共享额度，评分结束后释放。
API 缓存命中不占请求额度。轨迹 metadata 的 `concurrency` 保存实际峰值。
排队与执行分开：`queue_timeout_seconds: null` 默认不限制总排队时间；
`queue_stall_timeout_seconds: 1800` 只在同一资源连续 1,800 秒没有领取、释放或
真实工作进展时报告队列停滞。reset、生成、环境 step 和评分完成会报告进展，
不是用定时心跳掩盖卡住的任务。可显式设置有限的总排队上限，但长批次正常排队也
可能超过它。排队失败已有 trajectory ID，并保存 `failure_phase=trajectory_queue`、
等待时间、停滞时间及共享额度快照；不再笼统宣称 worker 故障。

用户 API 请求使用 `llm_args.timeout: 120`，Judge 请求默认也是 120 秒；
这不等于整条轨迹的墙钟超时（整条轨迹仍由轮数和 token 预算约束）。
worker 意外死亡时额度不会自动回收；取消时也要等后台工作真正结束才释放额度。
若确认卡死，需停止整个旧 Ray job 再恢复，避免孤儿环境仍在运行时超额启动。

评分只读 cleanup 前的 `environment_transcript`；`messages` 是裁剪后的 actor 上下文，
`token_turns` 是原始策略输出，不可混当沟通证据。Judge 单独失败保留同 ID/seed/slot 的
`scoring_inputs`，`--resume` 只重试评分；评分仍失败时不输出最终 pass@1/pass@4。

## 真实初始 prompt 长度预检

Parquet 的 `Initialize isolated Tau2 Airline task ...` 只是占位符，不能代表实际
上下文。运行时会完整编码 Policy、确认协议、工具定义、初始化得到的用户/工具消息，
保留所有 token，并在第一次生成前检查：

```text
initial_prompt_tokens <= rollout.initial_prompt_max_tokens（默认 8192）
initial_prompt_tokens + 1024 observation + 128 template + 64 generation <= 16384
```

超限不左截断，也不请求策略模型；轨迹 JSON 的 `metadata.initial_prompt` 保存
任务 ID、真实长度、预留量和失败原因。成功轨迹同样保存测量值。
`initial_prompt_max_tokens` 是训练/评估的数据 prompt 上限来源，训练时
`--extra data.max_prompt_length=...` 会同步到实际配置；不要先盲目提高上限。
后续 observation 仍按原预算截断内容，但不会再调用 veRL 的左截断辅助函数。

准备好固定 Tau2 环境、本地 tokenizer 和 RL 环境变量后，可先测 smoke 的初始化
（不加载 4B 权重、不生成 agent 回复、不调用 Judge；**会请求用户模拟器 API**）：

这里的 Judge 指本项目 DeepSeek 评分器。结束初始化环境时，Tau2 自身可能计算
内置奖励；预检不使用或报告这些分数。

```bash
python scripts/check_initial_prompts.py \
  --live-user-api --split smoke \
  --model "$MERGED_SFT_MODEL" --tau2-root "$TAU2_ROOT" \
  --output outputs/reports/initial_prompts_smoke.json
```

再测 `--split rl_train` 和 `--split internal_dev`。初始化 seed 与对应 Parquet
保持一致；报告记录 min/max/p50/p90/p95/p99、逐任务长度、失败与缺失任务。
这是该次真实初始化的测量，不保证未来不同 API 回复也一样长，运行时检查始终保留。
已有新版轨迹时可以完全离线汇总：

```bash
python scripts/check_initial_prompts.py \
  --records outputs/runs/rl_e2e_step1/trajectories --split smoke \
  --output outputs/reports/initial_prompts_observed.json
```

旧轨迹缺少初始测量时标记 `not_measured`，不会用整条轨迹长度冒充初始 prompt。
报告输出不覆盖已有文件；任一测量失败、超限或请求的任务缺失时返回非零退出码。

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
python scripts/verify_qwen_tool_roundtrip.py
python scripts/run_reward_audit.py data/audits/reward_audit.v1.json
```

第二条命令使用真实 Qwen tokenizer/chat template、veRL `hermes` parser 和
Tau2 的只读 `list_all_airports` 工具，必须得到恰好一个调用、执行成功且
数据库哈希不变。
该工具调用是人工构造再经模板渲染的，不是策略模型生成的，不能替代真实工具能力验收。

Stage 2，小规模 smoke：

```bash
python scripts/train_airline_grpo.py \
  --stage smoke --run-name rl_e2e_step1 \
  --tau2-root "$TAU2_ROOT" --verl-root "$VERL_ROOT" \
  --extra trainer.total_training_steps=1 \
  --extra trainer.save_freq=1 --extra trainer.test_freq=1 \
  --extra +trainer.ppo_audit=true
```

一次更新可能先生成 64–192 条候选轨迹；收不满组还可能再次尝试，另有训练前验证。
这不是单请求测试。先用默认超参数完成真实更新、checkpoint、导出和 internal-dev
评估闭环，不要直接运行正式 15 epoch，也不要用 official-test 反复联调：

```bash
python scripts/export_verl_lora.py --stage rl --verl-root "$VERL_ROOT" \
  --local-dir outputs/runs/rl_e2e_step1/checkpoints/global_step_1/actor \
  --target-dir /root/models/rl_e2e_step1_export
python scripts/evaluate_airline.py --split internal_dev --samples 4 \
  --tag rl_e2e_step1_dev --model-path "$MERGED_SFT_MODEL" \
  --lora-adapter /root/models/rl_e2e_step1_export/lora_adapter \
  --tau2-root "$TAU2_ROOT" --verl-root "$VERL_ROOT"
```

只有 checkpoint 实际保存成功后才执行导出。另需设计至少两次更新的独立运行，
在第一个 checkpoint 后中断并恢复，验证 optimizer、scheduler、计数和后续更新。
`ppo_audit` 目前检查整个 actor update 的前后边界，
`epoch_boundaries_instrumented=false`；它不等于已经验证 worker 内部两个 PPO epoch。
LoRA 梯度/参数变化、冻结底座不变及下一轮 vLLM 使用新权重，仍需 GPU 实测。

通过上述验收后，如需学习率对比，只在单独 run 中比较 `1e-5`：

```bash
python scripts/train_airline_grpo.py \
  --stage smoke --run-name smoke_lr1e-5_seed42 \
  --tau2-root "$TAU2_ROOT" --verl-root "$VERL_ROOT" \
  --extra actor_rollout_ref.actor.optim.lr=1e-5
```

Stage 3，24 条任务训练并只看 6 条 internal dev：

```bash
python scripts/train_airline_grpo.py \
  --stage internal_dev --epochs 15 \
  --run-name internal_dev_frozen_v1_seed42 \
  --tau2-root "$TAU2_ROOT" --verl-root "$VERL_ROOT"
```

Stage 4，冻结配置后使用全部官方 train：

```bash
python scripts/train_airline_grpo.py \
  --stage full_train --epochs 15 \
  --run-name full_train_frozen_v1_seed42 \
  --tau2-root "$TAU2_ROOT" --verl-root "$VERL_ROOT"
```

`full_train` 已包含原先 6 条 internal-dev，因此这时出现的 internal-dev
数值只能叫训练集监控指标，不能再叫独立验证结果。

如需继续中断的同一个实验，必须显式指定其 run name 和 checkpoint：

```bash
python scripts/train_airline_grpo.py \
  --stage internal_dev --run-name internal_dev_frozen_v1_seed42 \
  --resume-from-path outputs/runs/internal_dev_frozen_v1_seed42/checkpoints/global_step_N \
  --tau2-root "$TAU2_ROOT" --verl-root "$VERL_ROOT"
```

恢复也允许保存到另一个原本为空（或尚不存在）的新 run 目录；目标目录、底座
身份、目标已有配置的校验都发生在第一次写文件之前。另一个非空 run 仍会被拒绝。
`--resume-from-path` 无论是否换目录，都表示继续 checkpoint 的 optimizer/scheduler
和训练计数；**不等于仅加载模型权重开启新实验**。后者应先导出/合并所需模型，
设置新的 `MERGED_SFT_MODEL` 和 run name，不传 `--resume-from-path`。

Stage 5 前，先从 RL 检查点的 `actor/` 子目录导出标准 PEFT adapter：

```bash
python scripts/export_verl_lora.py \
  --stage rl --verl-root "$VERL_ROOT" \
  --local-dir outputs/runs/full_train_frozen_v1_seed42/checkpoints/global_step_N/actor \
  --target-dir /root/models/qwen3_4b_airline_rl_export
```

然后只在最终冻结后运行 test；不做动态采样或参数更新：

```bash
python scripts/evaluate_airline.py \
  --split official_test --samples 4 --tag sft_baseline \
  --model-path "$MERGED_SFT_MODEL" \
  --tau2-root "$TAU2_ROOT" --verl-root "$VERL_ROOT"

python scripts/evaluate_airline.py \
  --split official_test --samples 4 --tag sft_grpo \
  --model-path "$MERGED_SFT_MODEL" \
  --lora-adapter /root/models/qwen3_4b_airline_rl_export/lora_adapter \
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

候选尝试使用 `attempt_step`，真正更新使用 `optimizer_step`。跳过候选批时
只增加前者；学习率、epoch、终止条件和 checkpoint 仍绑定
`optimizer_step`。两者会原子写入每个 run 的 `step_counters.json`。

每个 checkpoint 内还保存自己的 `step_counters.json`，恢复时优先读取它并
验证 optimizer step；旧 run 级计数仅在恰好匹配该 checkpoint 时兼容。
训练 YAML 是所列超参数的来源，支持 `--config`；实际配置保存为
`runtime_config.yaml`，每次启动记录保存在 `launches/`。

这段扩展依赖 veRL v0.9.0 的受保护接口，因此训练启动器会先检查完整 commit SHA；版本不符会直接终止。

## 离线重新打分

当只调整 reward 权重、过程扣分或必须动作标注，而且 Judge rubric 未变时，
不必重新请求 DeepSeek。`reward`、`process_penalties`、软轮数和扣分上限会
从指定 YAML 显式构造成 `RewardConfig`，不会退回代码默认值：

新版会严格核对轨迹中的 Judge rubric 指纹。旧版笼统 Policy 判断不能用于
新版原子规则；需要重新 Judge。输出必须使用新的空目录，不能覆盖原轨迹。

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

Judge 缓存键包含模型、provider/base URL、完整消息、rubric/prompt/schema
版本、解码参数和 scorer 版本；缓存及轨迹均使用临时文件加原子替换。
未知工具与非法 Schema 不进入 Tau2，超长 observation 会在进入训练上下文
前确定性截断并记录。Tau2 的 timeout、max_steps、agent/user/infra error 等
终止原因会原样区分；基础设施失败轨迹保存审计信息后抛出，不进入 pass@k
分母。

# Tau2 Airline Agentic RL

> Current default: **v13-procredit-immediate-independent**. See [the current formula and compatibility notes](docs/reward_v13.md). Historical version descriptions below remain for audit.

这是本仓库的强化学习与冻结评估模块。项目使用固定版 Tau2 Airline 环境、选定的 Qwen3-4B SFT（或上一阶段 RL）合并模型和新挂载的 RL LoRA，在 veRL v0.9.0 上进行 GRPO / DAPO-style 训练。项目总览、数据版本与实验路线见 [根目录 README](../README.md)。

现行实验从 Airline SFT v4 的 Step 334 合并模型开始。SFT 数据、冻结官方基线、v13 离线验证与后续 GRPO 的状态见 [当前实验主线](../README.md#当前实验主线)。旧的两阶段 RL 和早期 ProCredit 消融见 [历史实验归档](../docs/legacy-experiments.md)。

`d10a8e7` 默认使用 `v12-procredit-35turn-zero-truncation`。PR #10 的 `70eee2e` 引入当轮进度增量及独立拒绝奖励 v13。两版均采用 [35 轮截断归零与训练重采](TRUNCATION_35.md)。转人工动态判断、确定性重复查询与 DB/COMM/REQ 进度见 [v11 改动](REWARD_V11.md)。

三套数值始终分开保存：

- `official_scores.reward`：Tau2 官方奖励；
- `custom_reward.strict_success`：本项目严格成功；
- `custom_reward.train_reward`：只用于 GRPO 的训练奖励。

`R_tau2_official != S_custom_strict != R_train`，新测评主指标为截断计零的 evaluation_pass1/pass4，同时单列 raw_tau2_official_pass1/pass4；旧口径按manifest保留。

最新复审修复、评估断点续跑、PPO 审计开关与尚未完成的 GPU 验收见
[REVIEW_FOLLOWUP.md](REVIEW_FOLLOWUP.md)。CPU 测试通过不代表 GPU 端到端验收已完成。

## 已实现的关键约束

- Tau2 固定提交：`a2c024725189473d2d7cea3a5cfdbcc67478e41f`；
- veRL 固定提交：`483b8a009ba3a97563edee3a19887e4862b8094a`（v0.9.0）；
- 每条 rollout 创建独立 `AgentGymEnv` 和 Airline DB；
- Qwen3-4B-Instruct 的 JSON `<tool_call>` 固定使用 veRL `hermes` 解析器；
- 同轮允许多个独立工具调用，逐个校验并按原顺序执行合法调用；未知工具和非法 Schema 只返回对应调用错误，非法 JSON 不进入 Tau2；
- RL/评估按实际配置逐字加载冻结系统提示词。写数据库前必须列出细节并取得用户明确确认，Agent Judge 按绑定政策检查。仓库默认绑定多工具版本；
- Qwen 原始生成 token ID、vLLM old log-prob 和 turn 边界原样保存；
- 只有 Qwen 输出 token 的 `response_mask=1`，工具、用户、模板 token 均为 0；
- 多轮 observation 包含 Qwen `turn_separator`，完整计入上下文预算；
- 16,384 token 生成前预算检查，训练与评估均为 35 个助手轮次上限，截断归零，训练截断最多重采 2 次；
- DeepSeek 用户模拟器与 Judge 使用独立 prompt、缓存和重试；
- 原 Judge 每条完整轨迹执行一次评分流程；默认新模式额外逐轮检查实际可见前缀，二者均缓存结果并有界重试；
- 官方 DB/COMMUNICATE、必须动作、语义与政策检查；strict 成功与训练奖励分别计算，默认 v4 将政策违规归因到对应轮次；
- GRPO 组大小 8、组内标准差归一化、Clip-Higher 0.20/0.28、dual clip 10；
- PPO epoch=2、old policy 固定为 vLLM rollout log-prob、token-mean、无 KL、无 Critic；
- RL LoRA `r=32, alpha=64, dropout=0, all-linear`；
- 自定义有界动态采样：首轮 4 个任务组、每组 8 个逻辑槽位（32 条），过滤后只补缺少的组，最多 3 轮（最坏 96 槽位，槽内实际补采另计）；每次更新最多使用 4 组，默认 ProCredit v4 保留异常耗尽组中的成功成员，上限处不足 4 组也继续更新；
- train/internal-dev/test 物理分离，test 标注与训练配置分开；
- 每条轨迹原子化保存，可离线重新打分。
- 每个实验使用独立 run 目录且默认禁用自动续训；
- 基础设施失败单独落盘，不计作模型失败或 pass^k 样本。
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

## SFT、RL 与评估的提示词和政策

已完成的 SFT v4 和冻结 SFT 基线使用单工具政策 `54c8a8153c0f9fe4d2bc4d57a3698221d09388d84245afdcc253378ee4238cb2`。以下说明针对仓库默认的多工具配置。实际启动与复现时，从该次运行的配置和 manifest 核对绑定。

`configs/prompts/areal_airline_sft.v1.json` 是当前冻结的系统提示词。
该文件的来源元数据为用户提供的 **8 条多工具样本**，不是对历史
10,649 条 AReaL 数据共有提示词的证明。当前绑定 SHA-256 为：

~~~text
e490e0859e1357ed6dd777fdbddab15b11d6de96ff7161328a6babcab77348ab
~~~

RL、冻结评估、初始长度预检和工具往返测试使用该文件，启动时按配置中的
`system_prompt_sha256` 验证内容。缺失、变化或配置缺少绑定时直接报错，
不会回退到 Tau2 官方政策。工具定义仍由 Qwen 的真实模板渲染。

当前政策要求：

- 同轮可以调用多个独立工具；有结果依赖的调用必须分轮执行。
- 调用工具时不能同时向用户发送正文回复。
- 更新预订数据库前，必须列出操作细节并取得用户明确同意。
- 不引入隐藏的 `action_proposal` 标签或额外确认协议。
- Judge 使用绑定提示词的 `<policy>` 正文；确认检查项为
  `database_write_confirmation`。旧 `confirmation_details` 名称已移除，
  不表示确认义务被删除。
- 旅行券等业务限制按当前绑定政策判定；旧版额外的
  `travel_certificate_new_booking_only` 检查不应冒充当前规则。

同轮各工具分别处理错误，合法调用可继续执行；多工具本身不算错误。
逐轮奖励在最终优势阶段应用坏轮规则。v10–v12 见 [逐轮奖励历史](SIMPLE_TURN_REWARD.md)，v13 见 [PR #10](https://github.com/3361119449-bit/agentic-rl/pull/10)。

原始历史 SFT JSONL 未因提示词绑定而被重写。更换 SFT 数据时，先检查真实
system 消息及数据哈希，再确认训练、RL 与评估的政策是否一致；不要把不同
提示词的数据混用后声称完全对齐。提取器拒绝混合 system 消息，输出必须不存在：

~~~bash
python scripts/export_sft_system_prompt.py --help
python scripts/export_sft_system_prompt.py --output /tmp/areal_sft_prompt_check.json
python -m pytest tests/test_sft_agent_policy.py tests/test_rollout_integration.py
~~~

上面的默认提取输入针对历史 AReaL 数据，输出仅用于核对，不会替换当前绑定；
检查新数据时按 `--help` 指定其输入。独立 `training/tau2_rollout_sft/` 流程
导出的上下文也不自动等于当前绑定版本。

提示词、奖励或 Judge rubric 变化均需新的实验身份与缓存。旧 checkpoint
仅在完整恢复身份一致时才能续训；用旧权重开启新条件时应先导出/合并并新建 run。
Tau2 官方任务、工具、数据库和评分算法不因这些项目政策检查改写。

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

- `MERGED_SFT_MODEL`：完整底座模型目录，当前实验选用 SFT v4 Step 334 合并模型；
- `DEEPSEEK_USER_MODEL`：服务商中用户模拟器的精确模型 ID；
- `DEEPSEEK_JUDGE_MODEL`：Agent 奖励 Judge 的精确模型 ID；训练与评估默认使用；评估显式传 `--no-reward-judge` 时不需要该奖励模型，但独立用户合规检查仍必填；
- `DEEPSEEK_USER_SIM_JUDGE_MODEL`：独立用户模拟器合规检查模型的精确 ID；训练和评估必填，可与奖励 Judge 使用同一型号，但提示词、输入与缓存独立；
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
  --project-config configs/rl/airline_procredit_v4.yaml \
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

当前默认训练配置 `configs/rl/airline_procredit_v4.yaml`、internal-dev/官方训练集
评估使用的 `configs/rl/airline_grpo_v1.yaml`，以及 official-test 配置
`configs/evaluation/airline_eval_v1.yaml`，其 `rollout` 中以下五项默认均为 **8**：

```yaml
agent_worker_count: 8
max_active_trajectories: 8
user_api_max_inflight: 8
judge_api_max_inflight: 8
vllm_max_num_seqs: 8
```

这五项分别控制 worker 数、活跃轨迹、用户 API、Judge API 与 vLLM 序列上限。
轨迹在创建环境前领取同一个 Ray job 的共享额度，评分结束后释放；API 缓存命中
不占请求额度。轨迹 metadata 的 `concurrency` 保存实际峰值。
8 是并发上限，实际峰值仍以 run 的 `runtime_config.yaml`、`launches/` 与轨迹
审计为准。历史 `airline_procredit_v1/v2/v3.yaml` 和 `airline_procredit_v4_legacy.yaml`
保留原有并发值，用于复现旧条件。并发变化也属于实验配置变化：已有 run 的
`runtime_config.yaml` 不会被仓库新默认值改写，新设置请使用新的 run/tag。
组大小、每批组数和补采上限另有训练契约校验，不能假定只改 YAML 就支持任意采样规模。
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
`scoring_inputs`，`--resume` 只重试评分；评分仍失败时不输出最终 pass^1/pass^4。

评估启动器对 `scoring_pending_records` 使用固定最多 **4 个 scoring task** 的
工作池，某个任务完成后才接下一条，不为全部待评分记录一次性创建任务。这里的
4 是评分任务上限；每次真正发出 Judge HTTP 请求（包括重试），仍须领取由运行时
`rollout.judge_api_max_inflight` 控制的共享额度。比如该配置为 2 时，可以有
4 个评分任务同时处理，但至多 2 个正在请求 Judge；配置大于 4 也不会让本批超过
4 个请求。缓存命中和重试退避不占 API 额度，不需要为了启用 4 个评分任务把 API
配置也改成 4。整个评分批完成后才重新检查覆盖率，评分失败仍保留原 ID/seed/slot，
不重采会话。取消或未预期异常会取消并等待其他工作任务结束，再退出该批。

这个改动只调整评分重试的调度，不改在线轨迹并发、奖励、提示词或 pass^k 口径；
已有 `--resume` 的代码身份校验仍然生效，不绕过它混续不同代码版本。

## 真实初始 prompt 长度预检

Parquet 的 `Initialize isolated Tau2 Airline task ...` 只是占位符，不能代表实际
上下文。运行时会完整编码 SFT 系统提示词、工具定义、初始化得到的用户/工具消息，
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

## 运行流程

### 本地检查与小规模 RL

先运行 CPU 不变量测试，再做真实 API/GPU 轨迹和人工 reward 审计：

```bash
python -m pytest
python scripts/verify_qwen_tool_roundtrip.py
python scripts/run_reward_audit.py data/audits/reward_audit.v1.json
```

CPU 回归包含全部 `ToolErrorKind` 的扣分覆盖，以及“普通文本 + 工具调用”从
格式校验到最终奖励的路径。旧过程成本对此错误为 0.10；默认新模式还将它作为
坏轮，在最终优势上应用 `min(A0, -1)`。该模型错误不应抛出评分异常或被当作
基础设施失败。另覆盖独立 Tau2→SFT 转换的超长过滤和 SFT 消融报告的
完整 test/trial 校验；真实 tokenizer 用例需要设置 `QWEN_TOKENIZER_PATH`。

第二条命令使用真实 Qwen tokenizer/chat template、veRL `hermes` parser 和
Tau2 的只读 `list_all_airports` 工具，必须得到恰好一个调用、执行成功且
数据库哈希不变。
该工具调用是人工构造再经模板渲染的，不是策略模型生成的，不能替代真实工具能力验收。

首次 GPU 更新使用小规模 smoke：

```bash
python scripts/train_airline_grpo.py \
  --stage smoke --run-name rl_e2e_step1 \
  --tau2-root "$TAU2_ROOT" --verl-root "$VERL_ROOT" \
  --extra trainer.total_training_steps=1 \
  --extra trainer.save_freq=1 --extra trainer.test_freq=1 \
  --extra +trainer.ppo_audit=true
```

一次更新先提交 32 个逻辑槽位；若 GRPO 过滤后不足 4 组，只补缺少的组，
最多 3 轮、最坏 96 个逻辑槽位。已有有效组不重采，实际交互补采另计，默认 v4 每槽最多
1 次初采、2 次交互异常替换及 2 次用户无效替换，另有训练前验证。
这不是单请求测试。先用默认超参数完成真实更新、checkpoint、导出和 internal-dev
评估闭环，再进入后续实验；不要用 official-test 反复联调：

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

### 从 SFT v4 开始 GRPO

将 `MERGED_SFT_MODEL` 指向 SFT v4 Step 334 的完整合并模型。选定奖励版本后，用新的 run 名称启动。v13 实验使用包含 PR #10 的代码，并核对实际配置中的 `reward_version: v13-procredit-immediate-independent`、`progress_credit: immediate_delta_v1` 与 `final_turn_rule: simple-turn-v2`。提示词和用户筛选以该 run 的记录为准。

`--stage internal_dev` 选择 24 个 RL train 任务和 6 个独立开发任务。按本次实验设置 `RL_EPOCHS` 和 `RL_UPDATES`，前者控制遍历上限，后者控制 optimizer 更新上限。正式运行前可加 `--dry-run` 检查命令。

```bash
python scripts/train_airline_grpo.py \
  --config configs/rl/airline_procredit_v4.yaml \
  --stage internal_dev --epochs "${RL_EPOCHS:?set epoch limit}" --seed 42 \
  --run-name sft_v4_grpo \
  --tau2-root "$TAU2_ROOT" --verl-root "$VERL_ROOT" \
  --extra "trainer.total_training_steps=${RL_UPDATES:?set update count}" \
  --extra trainer.save_freq=1 --extra trainer.test_freq=1
```

记录实际 `optimizer_step`、checkpoint、运行配置、模型身份和启动参数。v13 的 80 条离线重放复用了冻结轨迹和保存的判定，验证了优势计算；GPU 更新和训练后得分由后续 run 验证。

### 中断续训

继续同一个实验时才使用 `--resume-from-path`。保留原阶段、seed、epochs、
配置与全部训练 overrides。例如恢复上面的同一 run：

~~~bash
python scripts/train_airline_grpo.py \
  --config configs/rl/airline_procredit_v4.yaml \
  --stage internal_dev --epochs "${RL_EPOCHS:?set original epoch limit}" --seed 42 \
  --run-name sft_v4_grpo \
  --resume-from-path outputs/runs/sft_v4_grpo/checkpoints/global_step_N \
  --tau2-root "$TAU2_ROOT" --verl-root "$VERL_ROOT" \
  --extra "trainer.total_training_steps=${RL_UPDATES:?set original update count}" \
  --extra trainer.save_freq=1 --extra trainer.test_freq=1
~~~

恢复也允许保存到另一个原本为空（或尚不存在）的新 run 目录；目标目录、底座
身份、原 checkpoint 的训练身份和目标已有配置都在第一次写文件前校验。
另一个非空 run 仍会被拒绝。`rl_resume_identity.json` 保存于 run 目录及每个新
checkpoint 中，绑定阶段、训练/验证 Parquet 内容指纹、标注/AgentLoop 配置文件、
实际种子、epochs、总训练步数覆盖及其他训练 overrides。换目录不允许绕过校验。
`--extra` 不得改写 seed 或 epochs；应使用 `--seed`、`--epochs`。

旧 checkpoint 若缺少自己的 `rl_resume_identity.json`，不能直接恢复 optimizer /
dataloader 状态，也不能从当前配置或 run 目录补造身份。已有模型权重没有被删除，
仍可在满足原底座身份校验的前提下导出/合并后开启新实验；这不等于精确续训。
`--resume-from-path` 无论是否换目录，都表示继续 checkpoint 的 optimizer/scheduler
和训练计数；**不等于仅加载模型权重开启新实验**。后者应先导出/合并所需模型，
设置新的 `MERGED_SFT_MODEL` 和 run name，不传 `--resume-from-path`。

### 冻结评估

选择实际保存的 RL checkpoint，再从其 `actor/` 子目录导出标准
PEFT adapter。将下面的 `RL_CHECKPOINT` 设置为真实已保存的检查点目录：

```bash
python scripts/export_verl_lora.py \
  --stage rl --verl-root "$VERL_ROOT" \
  --local-dir "${RL_CHECKPOINT:?set frozen checkpoint directory}/actor" \
  --target-dir /root/models/qwen3_4b_airline_rl_export
```

评估 SFT 基线时，`SFT_BASELINE_MODEL` 必须指向原 SFT 完整模型；评估 RL
adapter 时，`RL_ADAPTER_BASE_MODEL` 必须指向该 adapter 训练时的同一底座。
正式对照先在运行 checkout 的 `configs/evaluation/airline_eval_v1.yaml` 中设置 `user_sim_filter.enabled: false`，并确认提示词路径和哈希已绑定到本次实验使用的版本。冻结两个模型后运行：

```bash
python scripts/evaluate_airline.py \
  --split official_test --samples 4 --no-reward-judge --tag sft_baseline \
  --model-path "${SFT_BASELINE_MODEL:?set original SFT model directory}" \
  --tau2-root "$TAU2_ROOT" --verl-root "$VERL_ROOT"

python scripts/evaluate_airline.py \
  --split official_test --samples 4 --no-reward-judge --tag sft_grpo \
  --model-path "${RL_ADAPTER_BASE_MODEL:?set adapter training base directory}" \
  --lora-adapter /root/models/qwen3_4b_airline_rl_export/lora_adapter \
  --tau2-root "$TAU2_ROOT" --verl-root "$VERL_ROOT"
```

只汇总 Tau2 pass^1 和 pass^4：

```bash
python scripts/summarize_evaluation.py \
  outputs/evaluations/sft_baseline/trajectories \
  --output outputs/reports/sft_baseline_pass1_pass4.json

python scripts/summarize_evaluation.py \
  outputs/evaluations/sft_grpo/trajectories \
  --output outputs/reports/sft_grpo_pass1_pass4.json
```

仓库原生入口默认开启用户筛选和 Agent 奖励 Judge。`--no-reward-judge` 只关闭 Agent 奖励 Judge，用户筛选使用前述 YAML 开关。官方对照使用完整工具回执、16K 上下文、8 并发和 35 个助手轮次，并记录实际运行配置。已完成的 SFT v4 基线使用 30 轮。严格对照时，SFT 和 RL 都按同一份配置评估。

评估使用 BF16 模型加载、参数和 rollout 精度，归约和缓冲使用 FP32。

两条评估路径与 `training/tau2_rollout_sft/report_pass1_pass4.py` 使用相同的数学公式：
每任务的 pass^k = `comb(成功次数, k) / comb(有效次数, k)`，最后对任务等权平均。
恰好运行 4 次时，pass^4 只有四次全部成功才为 1；只成功 1 次时 pass^1=0.25、
pass^4=0。它不是“至少一次成功”的 pass@4，定义见
[固定 Tau2 官方实现](https://github.com/sierra-research/tau2-bench/blob/a2c024725189473d2d7cea3a5cfdbcc67478e41f/src/tau2/metrics/agent_metrics.py)。

两份报告共用 `tau2_agentic_rl.pass_metrics` 的官方 ID 集合、版本校验和成功判定。
主线启动与离线汇总都会核对**准确的 20 个官方 test ID**及固定 Tau2 commit，
不只检查数量。官方 reward 和自定义 strict_success 必须是 `[0,1]` 内的有限数值；
`NaN`、无穷大、布尔值、字符串、缺失字段和越界值会报错，不会自动计作模型失败
或触发补采。正常基础设施失败记录允许评分为空，仍按原规则补齐缺失槽位。
官方成功判定逐字采用包含边界的 `1-1e-6 <= reward <= 1+1e-6`，
因此 `0.999999` 在主线和独立报告中都判成功。pass^k 公式没有改变。

同一评估身份下已有完整且合法的记录可以离线重新汇总，无需重训或重新生成轨迹。旧独立报告若
包含容差边界奖励，应重新汇总；缺少官方版本或评分无效的记录必须先核对原始证据，
不要补造身份字段或把异常分数改成零。评估运行的代码 hash 仍参与 `--resume` 校验：
更新代码后不能直接混续旧评估；未完成运行应保留原代码完成，或使用新 tag。

JSON 保留 `official_pass1` / `official_pass4` 等字段名，并新增
`metric_definition: tau2_pass_hat_k` 和 `metric_formula`。2026-09-07 此修复之前的
RL 汇总文件使用 pass@4，**不能直接与修复后的 pass^4 比较**。使用上面的汇总命令
从已有完整轨迹重新输出一份报告即可，不必重新 rollout 或请求 DeepSeek；不要只改旧报告标签。

## User Simulator 整条轨迹合规过滤（2026-09-12）

**训练和评估都支持独立 Agent Judge 开关**：`--reward-judge` / `--no-reward-judge`。
训练未传开关时遵循 YAML `judge.enabled`（默认 true），评估默认 true。
训练默认配置为 `configs/rl/airline_procredit_v4.yaml`。GRPO 不使用 critic，
因此非零 `trainer.critic_warmup` 会在启动时拒绝，避免虚假的 optimizer step 和 checkpoint。
训练关闭时 `reward_score` 与 DAPO 使用的 `train_reward` 都取审计后的官方奖励，
不计算原 Agent Judge 驱动的自定义复合奖励，也不伪造 `custom_reward` 或 strict 指标。
该分支仍保留既有官方截断处理（外部预算截断的官方 reward 计零），不会返回未打折的自定义奖励。
因此关闭 Judge 的 RL 属于**官方奖励消融**，要使用独立 run，不能中途切换后恢复旧 checkpoint。
用户合规过滤不受 Agent Judge 开关影响。例如：

```bash
python scripts/train_airline_grpo.py --stage internal_dev \
  --config configs/rl/airline_grpo_v1.yaml \
  --run-name official_reward_ablation --no-reward-judge --dry-run
# 正常训练：去掉 --dry-run；如需 Agent Judge，改为 --reward-judge。
```

v4 训练中，临时错误或 invalid user 补采耗尽只排除对应 slot，同组成功成员及其他有信号组继续训练。
配置、断言、token alignment 和未知代码错误通过 Ray job 共享的 fatal 标记传到 trainer，
阻止后续采样、补采和 actor 更新，不再被 veRL 的通用 `failure` 状态吞掉。

训练 GRPO / DAPO 与 `scripts/evaluate_airline.py` 共用 `user_sim_filter`，默认开启。
它在符合检查条件的完整交互结束后执行；按当前策略归零的截断轨迹跳过检查。
不修改 Agent 的 SFT 系统提示词，不给 User Simulator 追加纠错提示。
只因为 **User Simulator 本身实质破坏任务环境** 才丢弃整条 trajectory：

- `invented_fact`：编造 scenario、此前对话或 user tool 未提供的具体事实。
- `scenario_contradiction`：与场景明确的已知/未知信息、来电原因或任务指令矛盾。
- `goal_drift`：自行增加、删除或改变影响任务执行的目标/约束。
- `withheld_known_information`：被明确询问关键已知事实，却持续隐藏或答非所问。
- `conditional_behavior_violation`：提前执行或明显漏执行条件指令，并改变流程。
- `invalid_termination`：错误使用 STOP / TRANSFER / OUT-OF-SCOPE，导致错误终止。

Agent 失败、幻觉、政策违规、`official_reward=0` **绝不是过滤理由**。
用户相信 Agent 已完成后 STOP、用户提出违反 airline policy 的要求、啰嗦重复、
轻微 persona 或无实质影响的披露时机偏差均保留；外部截断本身不算用户错误终止。

检查 API 采用严格输入白名单，仅收到实际 User Simulation Guidelines（含运行时 persona）、
`user_scenario`、完整已发生对话和全部用户回复。不会收到 official reward、
evaluation criteria、参考动作、目标 DB 状态或标准答案；也不发送 Agent 系统提示词、
`raw_data` 或整份任务对象。实际对话中已经观察到的工具参数/结果保留。
落盘审计可以保留官方评分，但它与检查 API 输入分开，是否重采只取决于用户合规结论。

输出保存为 `user_sim_result`：`user_sim_valid`、`violation_type`、`severity`、
`turn_ids`、`reason`。有效时为 `true/none/none`，违规时为 `false/<上述类型>/hard`；
证据 ID 必须存在且指向用户回复。格式错误或不存在的证据触发同输入检查重试，
不会当作违规。`user_sim_inputs` 与其哈希冻结输入；传输前先原子保存待检查轨迹。

处理顺序是：完整交互 → 用户合规检查 → 合规才进入 Agent 奖励计算 / DAPO 分组
（Agent Judge 关闭时直接使用官方分数）。所有候选轨迹都检查，不先按奖励或 DAPO 组内方差挑选。
不合规轨迹留存审计，但其 token 不返回优化器，也不占有效 pass 样本；不是把它的奖励改成零。
训练在同一 policy version 下重新生成整条交互；评估由驱动补同一 task/sample slot，
换 seed、换独立用户缓存并记录尝试次数，避免命中原来的违规回复。

`user_sim_filter.max_resamples: 2` 表示原始尝试之外最多重采两次；到上限仍不合规时
令本条 rollout 报错/保留未完成评估，**不放行、不改成失败分、不无限重采本条**。
温度为 0 的服务仍可能再次产生同样回复，换 seed/缓存不保证成功。
检查 API 失败只重试同一冻结交互：评估 `--resume` 不会补采该 slot；训练的本条
rollout 报错并保留 pending 审计，不向 PPO 返回它。训练 replay buffer 沿用已有的
基础设施失败组处理：整组不参与优化，后续可以请求新的候选任务组；不会把该 pending
对话标成用户违规或在本条重采循环中重放。历史 pending 的核验用离线脚本，
不能把直接重跑训练当成已经完成了它的检查。
离线/评估检查最多 4 个 worker，实际 HTTP 请求仍共用 `judge_api_max_inflight`。

开启过滤后，报告是**以用户模拟器合规为条件的 pass^k**，公式和官方成功判定不变，
但样本集合与未过滤的原始 Tau2 评估不同。报告记录开关、拒绝数、原因和待检查 slot；
做消融时各组必须使用相同过滤模型/准则/重采上限，并披露这些统计。
独立旧报告脚本 `training/tau2_rollout_sft/report_pass1_pass4.py` 不执行该过滤，
不要把它的原始样本口径与新评估混用，也不要依据 test 得分调整过滤准则。
更改代码/过滤配置或 Judge 开关后使用新 run/tag，不能绕过旧身份校验。

已有磁盘轨迹也可逐条离线检查到新目录（不覆盖原文件）：

```bash
python scripts/screen_user_simulations.py \
  outputs/runs/my_run/trajectories outputs/user_screened/my_run \
  --config configs/rl/airline_grpo_v1.yaml --dry-run
# 确认输入后去掉 --dry-run，才会请求检查 API。
```

输出为 `accepted/`、`rejected/`、`pending/` 和 `screening_report.json`，含重采计划。
新记录已经保存指南；旧记录还必须提供 `--guidelines /path/to/exact_historical_guidelines.txt`，
并具有固定 Tau2 版本、原始场景及完整对话，缺失时拒绝猜测。
指南文件必须是当时实际使用的版本（含 persona 替换），不是随便取当前官方文件。
历史重采必须使用该记录的**原始 checkpoint/policy**；这个离线脚本不会擅自用当前模型
重采或将历史 `accepted/` 自动注入 on-policy DAPO。在线训练/评估的自动替换在上述
同 policy 的 actor/评估驱动完成。

本轮离线验证：Transformers 4.57.1 和 5.10.4 各 **444 passed / 1 skipped**，
包含真实本地 Qwen tokenizer、实际 actor/启动器的隔离执行、两种 Agent Judge 开关、
输入防泄漏、用户证据 ID、完整轨迹拒绝/重采、API pending 与评估 slot 补齐测试。
跳过项是旧的远程 tokenizer opt-in 测试；Ruff 和 `git diff --check` 通过。
没有启动 GPU 训练或真实付费 API，检查模型的语义误判率仍需在非 test 样本上人工抽查。

## 动态采样实现说明

veRL v0.9.0 的内置 V1 ReplayBuffer 会忽略 `algorithm.filter_groups.max_num_gen_batches`。本项目没有依赖该无上限行为，而是通过 `tau2_agentic_rl.verl_entrypoint` 构造 `CappedPPOTrainerSync`：

1. 同一 policy version 先生成 4 个 prompt group，共 32 个逻辑槽位；
2. 默认 v4 每组有 8 个逻辑槽位；无效 user 及明确可恢复异常只替换原槽位，policy 违规直接逐轮惩罚；
3. 槽位补采耗尽后，根据 worker 的终态成员记录保留成功评分的同组成员，用真实成员计算 advantage；致命错误立即停训；
4. 按最终 token advantage 是否有信号过滤组，不足目标 4 组时只补缺组，例如保留 3 组就只补 1 组；
5. 最多 3 轮，最坏 12 group / 96 个逻辑槽位；按实际提交的组数统计，不再用轮数乘固定批量。物理补采单独计数，默认 v4 最多 480 次交互尝试（另有验证）；
6. 每次更新最多使用 4 个有效组；到上限后只要仍有有效组，即使不足 4 组也更新，只有完全无信号才跳过；
7. PPO mini-batch 保持 32 行、两个 epoch。对齐所需的合成 padding 为零 loss mask，不参与组基线、训练奖励、有效轨迹统计或梯度；更新后再同步 vLLM 权重。

旧配置保留固定完整组与固定四组更新的语义。v4 不跨 policy version 缓存
多余组，也不复制正常轨迹来填满缺失的 group 成员。

候选尝试使用 `attempt_step`，真正更新使用 `optimizer_step`。跳过候选批时
只增加前者；学习率、epoch、终止条件和 checkpoint 仍绑定
`optimizer_step`。两者会原子写入每个 run 的 `step_counters.json`。

每个 checkpoint 内还保存自己的 `step_counters.json`，恢复时优先读取它并
验证 optimizer step；旧 run 级计数仅在恰好匹配该 checkpoint 时兼容。
训练 YAML 是所列超参数的来源，支持 `--config`；实际配置保存为
`runtime_config.yaml`，每次启动记录保存在 `launches/`；恢复另外核对原 checkpoint
中的 `rl_resume_identity.json`，不能只比较 YAML 而忽略 CLI 参数和数据内容。

这段扩展依赖 veRL v0.9.0 的受保护接口，因此训练启动器会先检查完整 commit SHA；版本不符会直接终止。

## 离线重新打分

早期 ProCredit 版本、可选 DB 子目标信用和完成后回退消融见 [历史实验归档](../docs/legacy-experiments.md)。

### 单轨迹重评分

启用逐轮奖励时，离线评分要求与配置匹配的逐轮语义标签。旧轨迹缺少标签时，纯离线脚本
会拒绝直接套用新模式，不会调用模型补标签；以下旧配置示例不代表新模式可无条件重放。

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

## 截断与 Judge 证据

采用 [35 轮截断归零](TRUNCATION_35.md)。训练与评估都保存实际终止原因和原始官方分。旧 `0.75` 折扣及当时的证据校验见 [历史实验归档](../docs/legacy-experiments.md#截断折扣与-judge-证据校验2026-09-12)。

## RL / 评估的 EOS 文本边界

Qwen 的生成 token 包含终止符 `<|im_end|>`。RL 与评估共用的 actor loop
在确认最后一个 token 确实是 EOS 后，只把它从**对外纯文本的解码输入**中排除。
发给 Tau2 User Simulator 的正文及 Judge 的实际会话记录不再包含这个模板终止符；
不是事后清洗 Judge transcript。系统提示词仍逐字使用当前冻结的多工具版本。

`raw_generated_text`、`token_turns`、返回 veRL 的 token ID、old log-prob 和
response mask 保留原始 EOS，下一轮仍接续原始 token 流。不会全局替换字符串、
删除正文中有意写出的字面标记或跳过所有特殊 token；工具解析和整轮格式校验仍
读取原始输出。只有 EOS / 空白的回复走本地 `parse_error` 反馈，不发送空动作给
用户模拟器，也不计作 API 故障。未生成 EOS 的截断回复仍不执行、不对外发送。
原始输出与被拒绝工具事件中的审计文本可能仍含 EOS，不能据此认定发生对外泄漏。

已把终止符发进 `environment_transcript` 的旧轨迹必须保留原样：用户模拟器的
后续反应可能已受影响，删字符串或只重跑 Judge **不能恢复干净的评估条件**。
正式 pass^1 / pass^4 应使用修复后的代码和**新的评估 tag**重新 rollout，不能
把新旧轨迹混合汇总或绕过 `--resume` 的代码身份校验。前文“可以离线重新汇总”
只适用于交互本身未受污染、仅统计口径需要修正的记录。

这个推理传输问题本身不要求重训 SFT 或重新导出已有正常 checkpoint；如果已用
受影响的 rollout 做过 RL 更新，应将其视为不同实验条件，不能声称修复代码后
旧训练结果就自动等价。更新仓库不会自动发起重评或训练。

## 正式训练前仍必须人工完成

- 记录实际使用的用户模拟器及 Judge 模型 ID；
- 在 A800 上完成一次端到端 工具交互验收，核对 Qwen3 工具标签与 tokenizer/chat template；
- 用真实 Airline 工具结果统计重新确认 1,024 token observation 预留量；
- 人工审计至少 50–100 条代表性轨迹及 Judge 结论；
- smoke 中记录训练侧 actor 与 vLLM rollout 的 ratio 分布（允许训推差异，不要求接近 1），核对 PPO 始终使用保存的 rollout 行为策略 log-prob，验证两个 PPO epoch 的 old log-prob 不变、checkpoint 可恢复且无 OOM；
- 在看任何 test reward 前冻结 reward、标注、prompt、训练步数和评估参数。

本地 CPU 测试不能替代上述 API、GPU 和人工验收。

Judge 缓存键包含模型、provider/base URL、完整消息、rubric/prompt/schema
版本、解码参数和 scorer 版本；缓存及轨迹均使用临时文件加原子替换。
未知工具与非法 Schema 不进入 Tau2，超长 observation 会在进入训练上下文
前确定性截断并记录。Tau2 的 timeout、max_steps、agent/user/infra error 等
终止原因会原样区分；基础设施失败轨迹保存审计信息后抛出，不进入 pass^k
分母。

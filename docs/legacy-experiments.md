# 历史实验路线与复现记录

本页归档仓库在 2026 年 11 月调整主线前的 AReaL 清洗、SFT 数据消融、M0 基线计划和两阶段 RL 流程。现行路线见 [项目首页](../README.md#当前实验主线)。

以下命令来自仓库 `d10a8e7` 及更早的实验说明。复现具体 run 时，以对应的代码提交、运行配置、数据哈希、模型身份、seed 和保存的 checkpoint 为准。示例里的 12 次更新是命令参数，历史实际步数以运行记录为准。

## SFT 数据与消融

早期使用清洗后的 AReaL 数据和官方 Tau2 train 成功轨迹，比较四组 SFT 条件：仅 AReaL、仅 Tau2、AReaL 后接 Tau2、AReaL 后接等步数的随机 AReaL。轨迹采样、16K 过滤、对照构建与当时的评估命令见 [完整消融流程](../training/tau2_rollout_sft/README.md)；训练入口见 [LoRA SFT 工具](../training/qwen3_4b_sft/README.md)。

## 历史 AReaL SFT 数据

路径：`datasets/tau2_airline_sft_strict_cleaned/`。

- 10,649 条 SFT 样本，来自 909 个源对话；
- 不含 `thinking` / `reasoning` 字段及 `<think>` 标签；
- 已按 Qwen3-4B 的 16K 上下文限制检查；
- 严格泄漏清理移除 90 个完整源对话、1,198 条前缀样本；
- 清理后审计未发现精确 test 参考查询、同日 test 目标航班暴露，或 tests 13、32、48 的高重叠保留模板。

这些审计结论针对该历史数据版本，不自动覆盖后续新采数据。完整流程与校验文件见 [数据清洗说明](../datasets/tau2_airline_sft_strict_cleaned/docs/CLEANING_PROCESS.md)。

数据 SHA-256：

~~~text
5535171a7b6f271d282966ffcb5d5b1a7a5581bc2aa7226667624d9e20ed4cb0
~~~

## 早期 M0 与 SFT 基线画像

M0 指未经本项目 SFT/RL 的原始 Qwen3-4B 模型；SFT 基线指完成监督微调后的
模型。当时 README 记录的 M0 实验安排为 24 个 `rl_train` 任务 × 4 次采样，必须核对
原始任务 ID、模型、提示词、seed、代码提交及轨迹清单。

当时的 `evaluate_airline.py` 仅接受 `internal_dev`、`official_train`、
`official_test`，没有 `rl_train` split 选项；不能将其 30 条官方训练任务的
默认画像命令标为 24 条任务 M0 复现命令，也不要以命令示例代替缺失的历史记录。

Stage -1 是保留的 **SFT 基线画像**（默认每个官方 train task 8 条，共 30 × 8）：

```bash
python scripts/profile_sft_baseline.py \
  --tau2-root "$TAU2_ROOT" --verl-root "$VERL_ROOT"
```

完成后会把组内方差、全 0/全 1、组件均值、工具错误、长度和逐任务统计写入 `outputs/reports/sft_baseline_profile.json`。该包装脚本继承当时的评估默认值，其中 Agent 奖励 Judge 开启。

多组实验可传入 `--tag areal_then_tau2`；评估轨迹目录和画像报告使用同一个
规范化后的 tag，避免误读默认目录的旧结果。传入 `--samples` 时画像分组大小
同步变化；`--dry-run` 只展示评估命令，不读取或覆盖已有画像。

## 旧两阶段 RL

默认 ProCredit 始终使用 24 个训练任务与 6 个独立 internal dev 任务。
`--stage internal_dev` 指该训练/验证组合；`--stage full_train` 会被拒绝。
旧的“15 epoch 后把全部 30 条任务并入正式训练”不再作为默认路线。

Stage 3，第一阶段从选定 SFT 合并模型新挂 LoRA。下面以最多 12 次 optimizer
更新为例，明确写出配置和步数；`--epochs` 是数据遍历上限，不是更新次数：

~~~bash
python scripts/train_airline_grpo.py \
  --config configs/rl/airline_procredit_v4.yaml \
  --stage internal_dev --epochs 2 --seed 42 \
  --run-name rl_stage1_simple_turn_seed42 \
  --tau2-root "$TAU2_ROOT" --verl-root "$VERL_ROOT" \
  --extra trainer.total_training_steps=12 \
  --extra trainer.save_freq=1 --extra trainer.test_freq=1
~~~

先加 `--dry-run` 检查命令。完成步数由实际 `optimizer_step` 与 checkpoint
确认，不能凭预设步数声明训练完成。上例为当时的奖励版本示例，不代表
此前服务器上的 12 步训练采用相同奖励、数据和代码。

Stage 4，从第一阶段 **实际已保存** 的 Step 12 导出 adapter，并合并到第一阶段
训练时的同一底座；合并脚本虽然名为 `merge_sft_lora.py`，也接受已导出的
标准 RL PEFT adapter，并验证其底座身份：

~~~bash
python scripts/export_verl_lora.py --stage rl --verl-root "$VERL_ROOT" \
  --local-dir outputs/runs/rl_stage1_simple_turn_seed42/checkpoints/global_step_12/actor \
  --target-dir /root/models/rl_stage1_step12_export

python scripts/merge_sft_lora.py \
  --sft-adapter /root/models/rl_stage1_step12_export/lora_adapter \
  --output /root/models/qwen3_4b_airline_rl_stage1_step12_merged

export MERGED_SFT_MODEL=/root/models/qwen3_4b_airline_rl_stage1_step12_merged
~~~

合并后按前述 adapter 等价性检查流程核对权重。第二阶段以此完整模型新挂 LoRA，
使用独立 run；先明确该实验的 epoch 上限与更新数，再执行：

~~~bash
# 必须先设置 RL_STAGE2_EPOCHS 和 RL_STAGE2_UPDATES。
python scripts/train_airline_grpo.py \
  --config configs/rl/airline_procredit_v4.yaml \
  --stage internal_dev --epochs "${RL_STAGE2_EPOCHS:?set stage-2 epoch limit}" --seed 42 \
  --run-name rl_stage2_simple_turn_seed42 \
  --tau2-root "$TAU2_ROOT" --verl-root "$VERL_ROOT" \
  --extra "trainer.total_training_steps=${RL_STAGE2_UPDATES:?set stage-2 update count}" \
  --extra trainer.save_freq=1 --extra trainer.test_freq=1
~~~

新阶段不传 `--resume-from-path`，optimizer、scheduler 和训练计数重新开始。
保留第一阶段来源、数据哈希、奖励版本、合并记录和每阶段实际更新数。服务器历史
第二阶段的确切更新数尚需由对应 run 记录确认，不能据此示例补造结果。

旧路线分别冻结 SFT、第一阶段和第二阶段的模型。评估一个 RL adapter 时，使用该 adapter 训练时的完整底座。第二阶段的底座是第一阶段合并模型。中断恢复的通用操作见 [续训说明](../agentic_rl/README.md#中断续训)。

## 历史奖励与评估口径

| 配置 | 保留的实验条件 |
| --- | --- |
| [ProCredit v1](../agentic_rl/configs/rl/airline_procredit_v1.yaml)、[v2](../agentic_rl/configs/rl/airline_procredit_v2.yaml)、[v3](../agentic_rl/configs/rl/airline_procredit_v3.yaml) | 早期过程信用、政策门控及归因消融 |
| [ProCredit v4 legacy](../agentic_rl/configs/rl/airline_procredit_v4_legacy.yaml) | 原始 v4 逐轮政策惩罚 |
| [v10](../agentic_rl/configs/rl/airline_procredit_v4_v10.yaml)、[v11](../agentic_rl/configs/rl/airline_procredit_v4_v11.yaml) | 旧逐轮拒绝去重与撤销规则，详见 [逐轮规则历史](../agentic_rl/SIMPLE_TURN_REWARD.md) 与 [v11 改动](../agentic_rl/REWARD_V11.md) |
| [v12 基线提交](https://github.com/3361119449-bit/agentic-rl/tree/d10a8e7622447b28df3021ae5ec80a7f6c855301) | 35 个助手轮次、截断归零，保留逐轮剩余回报与中心化 |

较早的复合奖励配置使用 `truncation_multiplier: 0.75`。该系数针对整条轨迹的终止原因，只作用一次。原始公式、证据校验与各版复现实例保存在下文。2026-09-07 修复之前的部分报告使用 pass@4，历史分数按各自 manifest 的口径保存。

### 可选官方完成后回退信用（2026-10-18）

`credit.post_completion_credit` 默认 `null`，设为 `post_completion_regression_v1`
后，依据原官方 DB/COMMUNICATE 前缀检查保留首次完成的正向信用，并将确认
破坏官方 DB 目标的轮次限制为最终 Token Advantage ≤ −1（叠加独立 process cost）。
同一错误已有 policy 惩罚时不再重复叠扣；与 DB 子目标局部正奖共享每条轨迹 0.5 上限。
不改变标量奖励、最终官方成功、GRPO 均值或采样设置。

对存在确认回退的轨迹，安全的首次完成轮在所有项合并后保底为
`max(1e-6, 0.5 − 当轮 process cost)`，确保最终 Advantage 为正。保底取最大值，
修正量单独审计，可超过局部奖励上限。policy 违规仍必须为负。
指定的 Task 12 T6 为 −1.7088 → +0.5，T10 的原惩罚不变。详见
[完成后回退规则与 738 条轨迹审计](../agentic_rl/POST_COMPLETION_REGRESSION_V1.md)。

### 可选 DB 子目标局部信用（2026-10-18）

v4 配置新增 `credit.db_subgoal_credit`，默认 `null`。设为
`db_subgoal_local_v1` 后，为同一订单中独立、已验证的航班、乘客或行李目标
增加当轮信用；`train_reward`、官方成功判定、policy/process、采样预算均保持原语义。
同轮包含错误、目标外或无法验证的写入时不给新增正奖；状态回退在破坏该目标的
轮次扣分，重复恢复不能刷分。不能确认独立性的取消/重订及 user-tool 任务不启用。

在修改后的 v4 YAML 中启用，再通过现有 `--config` 参数选择该文件。
切换信用条件需新 run，不绕过 optimizer resume 的配置/代码身份检查。
实测 738 条 PPO 轨迹、96 个原组：关闭开关完全复现旧 Token Advantage；
启用后 12 个正确轮次由非正变正，新增误奖励错误写入为 0。
完整规则、离线命令、真实轨迹对比及验证边界见
[DB 子目标局部信用与验收报告](../agentic_rl/DB_SUBGOAL_LOCAL_V1.md)。

### ProCredit v4 逐轮政策惩罚（2026-09-29）

本节描述原始 v4，现保留在 `configs/rl/airline_procredit_v4_legacy.yaml`。
后来 v10–v12 在主配置中加入 `simple-turn-v1` 最终优势规则，见 [历史规则](../agentic_rl/SIMPLE_TURN_REWARD.md)。
原始 v4 的训练奖励取消 policy/safety 整轨迹门控，
违规 turn 的局部奖励固定为 -1，叠加 trajectory advantage 后仍保证该 turn 的
最终 advantage ≤ -1。process 只在本轮按固定值扣分，取消整轨迹扣分与 cap，
与同轮 policy 相加；task/progress 不受 policy clipping。交互异常只补采原槽位，
Judge 异常只重评冻结轨迹；
临时错误重试耗尽后利用同组已评分成员；致命错误立即停训。多余有效组及
补采上限处的剩余有效组均用于当前更新。显式 BF16 延续 v3。
v4 的训练任务选择使用 `task_or_local_credit`，保留已授权训练集中的任务，
不再因为初始 progress 为零排除拒绝、政策推理等任务；实际无 advantage 信号的组仍过滤。
训练内验证也只补采失败 slot，耗尽后保留成功样本并报告
`val/coverage/{requested_slots,completed_slots,failed_slots,completion_rate}`；
全部失败时只报告覆盖率，继续训练。Tau2 后台线程保留原始异常，Judge 检查按 ID 匹配，
接受合法乱序，缺失、重复、额外 ID 和错误证据仍拒绝。
同轮多工具调用逐个校验：schema 无效或未知工具只返回对应调用的错误，
合法调用按原顺序执行，每个 call ID 都保留工具结果；不再整批阻断。
进度和最终评分的严格回放识别适配器记录的拒绝调用，按未执行处理；
其余调用继续严格核对，原始轨迹、Judge 证据和 token/log-prob 保留。
详见 [公式、重试边界和启动方式](../agentic_rl/REWARD_TURN_LOCAL_V4.md)。

### ProCredit v3 contract 修复（2026-09-28）

新增 `configs/rl/airline_procredit_v3.yaml`：canonical matcher 与 safety 解耦、
Judge applicability、DB/REQ progress 去重、零分轨迹局部过程惩罚、训练任务
信号筛选，以及显式 BF16。须启动新 run；详见
[行为、启动命令和验证边界](../agentic_rl/REWARD_CONTRACTS_V3.md)。

### ProCredit 新奖励实验（2026-09-27）

独立配置 `configs/rl/airline_procredit_v1.yaml` 启用 `strict_progress_v1`。
终局分数为 `S = V × max(0, m × (R + 0.5 × Phi_T) - P)`，范围 `[0, 1.5]`；
截断时 `m=0.75`，过程扣分在乘数之外。Phi 使用初始未满足的 DB、逐条 COMM、
逐条 REQ 检查，DB 可以回退。每轮只使用当时已送达环境的前缀，清理消息、
被拒绝的工具文本和未送达的截断输出不能贡献进度。

新训练器把组内归一化的轨迹 advantage 与逐轮剩余任务回报的中心化值相加。
逐轮中心只计算 gate 通过且有策略 token 的轮次；观测与 padding 的 advantage 为 0。
动态采样按最终 token advantage 是否全零筛组，因此终局同分但逐轮仍有信号的组
可以保留。每组 8 条、每次更新 4 组；最多 3 批、每批最多 4 组，
最坏 96 个逻辑槽位，槽内实际替换另计。

沿用前文依赖、模型和 API 环境变量设置，在 `agentic_rl` 目录运行：

```bash
python scripts/train_airline_grpo.py \
  --config configs/rl/airline_procredit_v1.yaml \
  --stage smoke --run-name procredit_smoke_seed42 \
  --tau2-root "$TAU2_ROOT" --verl-root "$VERL_ROOT"

# 完成 smoke 验收后，用 24 条训练任务和 6 条独立开发任务：
python scripts/train_airline_grpo.py \
  --config configs/rl/airline_procredit_v1.yaml \
  --stage internal_dev --epochs 15 --run-name procredit_v1_seed42 \
  --tau2-root "$TAU2_ROOT" --verl-root "$VERL_ROOT"
```

新模式要求 Agent Judge 开启，禁止会合并 30 条任务的 `full_train`。
旧奖励配置及官方评估口径保持原有行为。新模式轨迹使用 schema `2.0`，记录
`progress_trace`、`response_turn_ids` 和冻结前缀输入；run 下的 `group_audits/`
保存真实采样 uid、原始 8 个成员、计算输入、advantage 与过滤决定。
恢复身份绑定奖励/credit 配置、任务划分和项目 Python 代码内容；新旧奖励的
checkpoint 不能作为同一实验精确续训，修改实现后也应开启新实验。

仅比较逐轮消融时，可复制新 YAML，把 `credit.turn_coefficient` 改为 `0`，
并从原始组审计重算，输出必须是另外一个空目录：

```bash
python scripts/rescore_procredit_groups.py \
  outputs/runs/procredit_v1_seed42/group_audits outputs/procredit_ablation_groups \
  --config /path/to/procredit_lambda0.yaml
```

下方单轨迹重评分脚本也支持新配置，但必须已有完整的新模式 progress trace。
如果改动任务检查或动作依赖，旧进度会被拒绝，需先从冻结前缀重新验证。
组重评分的 `--records-dir` 可读取同一批成员的新评分记录；缺失成员、成员身份
变化或修改原始组审计都会报错。历史数据仅供离线分析，不会进入当前策略更新。

无模型/无 API 的真实依赖验收命令为 `python -m pytest contract_tests -v`，
需同时安装固定版本 veRL 和 Tau2，并设置 Tau2 数据路径；Tau2 未安装时其检查跳过。
本次本地验证覆盖 CPU 算法、生产循环和真实 PyTorch/TensorDict 张量，
完整固定依赖栈、真实 API 和 GPU optimizer 更新仍需在训练环境验收。

AgentLoop 会累计各轮服务端返回的 `min_global_steps` / `max_global_steps`，
并将整数范围传给 veRL 队列及 off-policy 指标。缺少范围时，仅使用服务端返回的
实际权重版本 `global_steps`；不能使用 prompt 的调度步号或默认 `policy_version=0`
替代。完全缺失版本时沿用 `policy_version_alignment` 致命错误通道，保留诊断，
不触发补采，也不在 PPO 更新之后才由指标统计报错。

可在无 GPU 环境用固定 veRL checkout 运行版本传递回归：

```bash
PYTHONPATH=src:tests:. VERL_VERSION_CONTRACT_SOURCE=/path/to/verl \
  python -m pytest tests/test_rollout_versions.py tests/test_verl_version_metrics.py -v
```

该回归校验固定源码哈希，并执行上游完整的队列后处理与 `_compute_metrics()` 方法，
覆盖权重版本 0、1 和服务端单版本字段的兼容路径；传输、张量运算和无关指标使用
CPU 替身。因此它不代表完整训练器/GPU smoke 已通过。正式训练前仍需在训练环境
完成短程 smoke，确认更新、指标、rollout 保存和日志全部完成。

### ProCredit v2：保留违规轨迹中的局部信号（2026-09-28）

`configs/rl/airline_procredit_v2.yaml` 保留终局 hard gate，新增对已定位违规 turn 的
局部负 credit，并允许归因完整轨迹中已验证的早期进度参与逐轮回报。违规动作
产生的正进度被屏蔽；旧 Judge 的支持性 `evidence_turn_ids` 不被猜作违规轮次。
无法定位的 policy 失败暂不产生正进度 credit，但已定位的违规仍受惩罚。

Judge 的 transfer 执行事实改由工具记录决定：没有成功 transfer 时强制
`valid=false`，不因矛盾的 transfer 字段丢弃整条轨迹，原始响应仍保留用于审计。
其他有效性检查继续执行。prompt/scorer 已更新，不能直接复用旧缓存身份。

沿用上方环境设置，用新 run name 启动：

```bash
python scripts/train_airline_grpo.py \
  --config configs/rl/airline_procredit_v2.yaml \
  --stage smoke --run-name procredit_v2_smoke_seed42 \
  --tau2-root "$TAU2_ROOT" --verl-root "$VERL_ROOT"
```

v1 的 credit 数学和组审计兼容性保留。v2 是新训练条件，不绕过旧 checkpoint 的
恢复身份检查。提供的旧 smoke 完整组在离线诊断中由无信号变为有负信号，
八条终局分数仍为零；这不代表真实 GPU smoke 已通过。数学定义、归因边界和
诊断计数见 [v2 设计说明](../docs/superpowers/specs/2026-09-28-policy-local-credit.md)。

## 截断折扣与 Judge 证据校验（2026-09-12）

本节保留 `v3-truncation-evidence` 的历史实现。各配置的具体公式以当时的 YAML 及版本文档为准。
以下是原复合奖励路径的截断处理说明，不能替代默认 v4 的逐轮过程成本与最终优势规则。
训练和评估配置均显式设置
`reward.truncation_multiplier: 0.75`，普通任务、转人工两个分支一致：

```text
最终 train_reward = 原有加权、过程扣分、裁剪及政策/安全门控后的奖励 × 截断系数
截断系数 = 0.75（轨迹被截断）或 1.0（其他终止情况）
```

判定依据是会话的 `termination_reason`，包含 `generation_truncated`、
`budget_exhausted`、`hard_turn_limit`、`max_steps`、`context_window_exceeded`。
单个 observation 被裁短但会话正常结束，不算整个轨迹截断；API 超时或其他
基础设施错误沿用原失败处理，不用折扣代替重试。折扣只乘一次，不改各分项、
`strict_success`、官方评分或 pass^k 公式。审计字段记录在 `custom_reward.details`
中的 `trajectory_truncated`、`termination_reason`、`reward_before_truncation`
和 `truncation_multiplier`。在线评分、冻结轨迹重试、离线重评分使用同一实现，
重复离线计算不会把旧奖励再次乘 0.75。

Judge 的 `semantic_checks`、`transfer_semantic_checks`、`mandatory_policy_checks`
及 `transfer_check` 的所有 `evidence_turn_ids` 必须是严格非负整数，并出现在
**实际传给 Judge 的** `messages.turn_idx` 或 `tool_events.turn_id` 中。
前者是环境会话编号，后者是 assistant 生成轮编号，可能包含未送进环境的被拒绝
工具尝试；这是两个证据视图，不强行当成同一消息编号。不会通过数组下标、数字
字符串或布尔值补造合法引用。空列表不新增“必须有证据”的要求，但原有的
“失败的 policy 条目必须有证据及具体理由”继续保留。

这个检查只验证 ID 存在，**不证明证据在语义上支持判定**。模型响应引用不存在的
ID 时按 Judge 格式错误有界重试，耗尽后保持评分失败待处理，不改成模型零分或
重采用户会话。缓存读取和离线重评分也会拒绝无效引用，不自动删除或修补证据。
Judge prompt / scorer 缓存版本已更新（v6 / v3）；旧 Judge rubric 指纹不兼容时
离线重评分会要求重新 Judge，不能伪造新版指纹绕过检查。

这次不改 SFT 数据、agent 系统提示词、工具规则或官方评分。新旧 reward / Judge
版本应使用分开的实验与评估 tag，不绕过恢复身份校验。未自动启动 GPU 或 API 作业。

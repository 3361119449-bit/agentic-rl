# agentic-rl

> Current default: **v13-procredit-immediate-independent**. See [the current formula and compatibility notes](agentic_rl/docs/reward_v13.md). Historical version descriptions below remain for audit.

围绕 **Tau2 Airline** 的 **Qwen3-4B 智能体训练与评估实验项目**，覆盖 SFT 数据清洗、轨迹采集与转换、LoRA 监督微调、GRPO / DAPO-style 强化学习，以及冻结模型的官方 `pass^1` / `pass^4` 评估。

项目重点是多轮工具交互、政策遵循和可追溯的训练信号：保存原始生成 token、行为策略 log-prob、工具执行记录、奖励证据与模型身份，使训练和评估可以审计。RL 运行基于固定版本的 Tau2 和 veRL；完整环境与命令见 [RL 使用说明](agentic_rl/README.md)。

## 项目组成

| 模块 | 内容 | 使用入口 |
| --- | --- | --- |
| AReaL 数据清洗 | 历史 Airline SFT 数据、泄漏清理、校验与来源记录 | [清洗流程](datasets/tau2_airline_sft_strict_cleaned/docs/CLEANING_PROCESS.md) |
| Tau2 轨迹与 SFT 数据 | 官方 train 任务采集、成功轨迹过滤、16K 转换与 SFT 消融 | [轨迹数据流程](training/tau2_rollout_sft/README.md) |
| Qwen3-4B SFT | veRL LoRA 微调、按源对话划分、adapter 导出与合并 | [SFT 训练说明](training/qwen3_4b_sft/README.md) |
| Agentic RL | 独立环境、过程信用、逐轮奖励、有界动态采样和 checkpoint 审计 | [RL 训练说明](agentic_rl/README.md) |
| 冻结评估 | 固定模型与样本槽位、失败恢复、官方指标与附加合规指标 | [评估命令](agentic_rl/README.md#分阶段运行) |

## 当前实现与默认行为

以下说明以 2026-11-02 的仓库实现为准；服务器上的运行条件需另查实际提交与运行记录。

- **多工具调用**：同一轮允许多个独立工具调用，逐个校验并按原顺序执行合法调用；依赖前一个结果的调用应分轮完成。工具调用与用户可见回复不能混在同一轮。
- **写数据库前确认**：当前绑定政策要求先列出操作细节并取得用户明确同意。Agent Judge 按该政策检查，不额外要求 `action_proposal` 标签。
- **固定数据边界**：官方 train 共 30 个任务，分为 24 个 RL train 和 6 个独立 internal dev；official test 为 20 个任务。默认 ProCredit 禁止 `--stage full_train`，避免把验证任务并入训练。
- **奖励 Judge 默认开启**：RL 训练与独立评估默认开启 Agent 奖励 Judge。评估可传 `--no-reward-judge` 仅计算官方指标；独立的 User Simulator 合规检查仍然开启并产生 API 调用。
- **采样规模**：每组 8 个逻辑槽位，每次更新最多使用 4 组；最多 3 批、每批最多 4 组，最坏 96 个逻辑槽位，异常与无效用户轨迹的替换另计。
- **并发默认值为 8**：当前训练与评估配置中的 worker、活跃轨迹、用户 API、Judge API 和 vLLM 序列额度均设为 8。实际并发以运行时配置、启动参数和轨迹审计为准；历史 ProCredit 配置保留原值。
- **提示词按内容绑定**：RL 与评估使用 [当前冻结提示词](agentic_rl/configs/prompts/areal_airline_sft.v1.json)，并检查哈希。该版本来源为用户提供的 8 条多工具样本，不能据此声称与全部历史或新采 SFT 数据一致。

### 默认逐轮奖励

默认 `agentic_rl/configs/rl/airline_procredit_v4.yaml` 已启用 `simple-turn-v1`，奖励版本为 `v12-procredit-35turn-zero-truncation`。35轮上限、截断归零与训练最多2次截断重采见 [截断策略](agentic_rl/TRUNCATION_35.md)。转人工动态判定、重复查询坏轮和 DB/COMM/REQ 加和进度见 [v11 说明](agentic_rl/REWARD_V11.md)。此前 v10 默认配置保存在 `airline_procredit_v4_v10.yaml`。在原 ProCredit v4 最终 token advantage `A0` 之后应用：

~~~python
if bad_turn:
    advantage = min(A0, -1.0)
elif eligible_correct_refusal:
    advantage = max(A0 + 1.0, 1.0)
else:
    advantage = A0
~~~

同轮任一明确工具错误或经证据确认的语义错误可触发整轮负向规则；正确拒绝按请求去重并检查后续撤销条件。坏轮优先。语义标签来自只读取当时可见前缀的额外 Judge 请求，增加训练 API 成本。

这项更新保留原有 ProCredit 的轨迹项、局部项和中心化，只在末端约束最终优势；不能将其描述为已经移除旧过程信用计算。轨迹总分与官方成功判定不因该末端规则改写。

规则、证据、缓存、命令与验证边界见 [简单逐轮奖励](agentic_rl/SIMPLE_TURN_REWARD.md)。原 v4 配置保存在 `airline_procredit_v4_legacy.yaml`，其他历史配置也保留。

## 实验路线与复现边界

1. **M0 原始模型基线**：与 SFT 模型基线分别记录。当前待整理的 M0 实验为 24 个 RL train 任务 × 4 次采样；它不是 `profile_sft_baseline.py` 默认的 30 个官方 train 任务 × 8 次画像。以原始模型、任务 ID、提示词、seed 和轨迹清单核对结果。
2. **SFT 数据采集与审核**：现有转换器提供成功分数、终止类型、工具错误和 16K 长度等基础过滤。新采轨迹仍需逐任务核对原始目标、转人工是否恰当、是否提前停止及政策是否一致；完整人工审核记录和新数据版本尚需随实验补齐。
3. **SFT 训练**：历史 AReaL 数据继续可用；新数据通过 `--data-jsonl` 显式指定。记录数据哈希、审核结论、提示词版本和输出模型身份，不把新数据误写成已替换默认输入。
4. **第一阶段 RL**：从选定 SFT 合并模型新挂 RL LoRA。12 次更新的命令示例见 RL 文档；实际完成次数以 checkpoint 与 `optimizer_step` 为准，不能把 epoch 数当更新次数。
5. **第二阶段 RL**：导出并合并第一阶段 Step 12 的 adapter，再以该完整模型新挂 LoRA、新建 run。它是从已有权重开始的新实验；精确恢复 optimizer 的 `--resume-from-path` 用于中断续训。
6. **冻结评估**：开发过程使用 6 个 internal dev 任务；固定模型、奖励、提示词和评估参数后再运行 official test。每个测试任务 4 条有效轨迹，共 80 条，报告 `pass^1` / `pass^4`。

两阶段的实际数据版本、奖励配置、第二阶段更新数和服务器修复同步情况，需要由对应 run 的 `runtime_config.yaml`、`launches/`、模型来源记录及代码提交确认。这里不将实验安排写成已完成的训练结果。

## 评估指标

始终区分三类数值：

| 字段 | 含义 |
| --- | --- |
| `official_scores.reward` | Tau2 官方奖励 |
| `custom_reward.strict_success` | 本项目附加的严格合规成功判定 |
| `custom_reward.train_reward` | 用于训练的自定义标量奖励 |

当前截断计零策略主指标为 `evaluation_pass1` / `evaluation_pass4`，原始 Tau2 指标单列为 `raw_tau2_official_pass1` / `raw_tau2_official_pass4`。开启 Judge 时另有 `custom_strict_pass1/pass4`；旧manifest继续保留原报告口径。

`pass^k` 衡量重复运行的一致成功率：每任务取 `comb(成功次数, k) / comb(有效次数, k)`，再对任务等权平均。4 次运行时，`pass^4` 要求 4 次全部成功。用户模拟器合规过滤默认开启，因此报告需披露过滤条件、拒绝数和补采情况；独立旧 SFT 报告未执行同一过滤，比较时需统一口径。

## 历史 AReaL SFT 数据

路径：`datasets/tau2_airline_sft_strict_cleaned/`。

- 10,649 条 SFT 样本，来自 909 个源对话；
- 不含 `thinking` / `reasoning` 字段及 `<think>` 标签；
- 已按 Qwen3-4B 的 16K 上下文限制检查；
- 严格泄漏清理移除 90 个完整源对话、1,198 条前缀样本；
- 清理后审计未发现精确 test 参考查询、同日 test 目标航班暴露，或 tests 13、32、48 的高重叠保留模板。

这些审计结论针对该历史数据版本，不自动覆盖后续新采数据。完整流程与校验文件见 [数据清洗说明](datasets/tau2_airline_sft_strict_cleaned/docs/CLEANING_PROCESS.md)。

数据 SHA-256：

~~~text
5535171a7b6f271d282966ffcb5d5b1a7a5581bc2aa7226667624d9e20ed4cb0
~~~

## 获取仓库

JSONL 使用 Git LFS 保存：

~~~bash
git lfs install
git clone https://github.com/3361119449-bit/agentic-rl.git
cd agentic-rl
git lfs pull
~~~

SFT 与 RL 使用不同的固定 veRL 版本，请按各自文档建立独立环境。配置 API 时使用 [环境变量示例](agentic_rl/.env.example)，精确模型 ID 与真实密钥由运行者填写。

## 验证与当前限制

仓库包含 CPU 回归、真实依赖契约、奖励审计与工具往返检查。CPU 检查覆盖不能替代 GPU 更新、真实 API 轨迹和人工语义审核；当前逐轮奖励文档也没有声称已通过新的 GPU 训练验证。具体待验收项见 [复审记录](agentic_rl/REVIEW_FOLLOWUP.md) 与 [RL 验收清单](agentic_rl/README.md#正式训练前仍必须人工完成)。

本仓库不对上游数据集或代码另行授予许可；使用时请遵守原始来源的条款。

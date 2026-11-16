# agentic-rl

> 当前主线：**SFT v4 Step 334 → 修复后的 30 任务 GRPO → 冻结官方评估**。正式配置为 [`airline_procredit_v4_train30.yaml`](agentic_rl/configs/rl/airline_procredit_v4_train30.yaml)，奖励使用 **v13.2-procredit-official-semantic-progress**。

围绕 **Tau2 Airline** 的 **Qwen3-4B 智能体训练与评估实验项目**，覆盖 SFT 数据清洗、轨迹采集与转换、LoRA 监督微调、GRPO / DAPO-style 强化学习，以及冻结模型的官方 `pass^1` / `pass^4` 评估。

## 当前实验主线

**审核后的 Airline SFT v4 → 下一助手轮 SFT → 冻结 Step 334 → 修复任务与 Judge 验证 → 30 任务 GRPO → 冻结官方评估。** 当前服务器准备、参数与启动方式见 [当前实验配置](agentic_rl/docs/current_experiment.md)。

1. **SFT 数据和训练已完成。** 使用审核后的 `20261106_full_semantic_v4` 数据，共 119 条轨迹、1,374 个目标轮，覆盖官方 train 的 30 个任务。按完整源对话划分为训练 1,338 行、验证 36 行。输入包含完整可见历史，loss 只监督下一次助手正文或工具调用。Qwen3-4B-Instruct-2507 使用 16K、non-thinking、LoRA rank 64、alpha 128、学习率 `1e-4`、训练 2 个 epoch。冻结合并模型来自 Step 334。[SFT 工具与格式](training/qwen3_4b_sft/README.md)
2. **冻结 SFT 官方基线已完成。** 在官方 test 20 个任务上各跑 4 条，共 80 条，8 并发，关闭用户合规筛选和 Agent 奖励 Judge，保留 Tau2 官方评分。该次使用 30 个助手轮次上限和完整工具回执，得到 **pass^1 42.5%、pass^4 20.0%**。这组分数使用当次 manifest 的口径。
3. **v13 奖励已完成离线验证。** 用已冻结的 80 条、1,227 轮复算。整轨迹 GRPO 仍按 `R = 终局成功 + 0.5 × 最终进度` 在组内归一化。当轮额外信用使用 `0.5 × (本轮后进度 − 本轮前进度)`，不再使用剩余回报及逐轮均值。同一请求在不同轮被正确拒绝可分别获奖，同轮有错则坏轮优先。该历史验算使用全局与逐轮 Judge；当前正式入口已合为一次整轨请求返回两类结论。[v13 实现与验证（PR #10）](https://github.com/3361119449-bit/agentic-rl/pull/10)
4. **当前正式 RL 使用全部 30 个训练任务。** 从 Step 334 的 bf16 合并模型新挂 RL LoRA，加载 Task 7、14、21、23、39 的修复。组归一化使用「Tau2 官方奖励 + 0.5 × 最终进度」，最终进度同时计入必要动作和整轨语义检查。启用整轨迹 Judge 与 30 任务指导，一次请求返回整轨结论及全部 `simple-turn-v2` 轮次标签。原来的 6 个内部验证任务已编入训练，关闭训练前及周期内部验证；官方 test 20 个任务继续隔离。每组 8 条、8 并发、最多 35 个助手轮次。[奖励公式](agentic_rl/docs/reward_v13.md#official-completion-and-terminal-progress) · [当前划分](agentic_rl/data/splits/airline_repaired_train30.v1.json) · [任务修复](agentic_rl/task_repairs/README.md)
5. **冻结模型并做同条件对照。** 正式 test 固定 20 × 4 = 80 个样本。新实验采用 35 个助手轮次上限，真实截断整条归零并记录原因。训练截断槽位最多重采 2 次，评估截断保留在分母。官方对照关闭用户合规筛选和 Agent 奖励 Judge，报告 `evaluation_pass1/pass4` 并单列原始 Tau2 分。与 30 轮基线做严格对比时，两边须使用同一提示词、轮次及评分口径。[截断策略](agentic_rl/TRUNCATION_35.md) · [冻结评估](agentic_rl/README.md#冻结评估)

### 实验配置与仓库默认值

正式 RL 和官方测评沿用 SFT v4 的单工具提示词 `54c8a815…`，上下文为 24,576 tokens。训练关闭用户合规筛选，官方测评同时关闭 Agent 奖励 Judge。启动入口使用 [正式配置清单](agentic_rl/configs/current.json)。

训练入口默认采用修复后的 `full_train`。复现旧划分时需通过 `--config` 显式选择归档配置并指定相应阶段。[奖励公式](agentic_rl/docs/reward_v13.md) · [训练命令](agentic_rl/README.md#从-sft-v4-开始-grpo)

## 历史实验

旧路线包括 AReaL 清洗、SFT 消融、24/6 RL 开发划分和两阶段 RL。数据版本、命令及历代奖励口径保留在 [历史实验归档](docs/legacy-experiments.md)。

## 指标

| 指标 | 用途 |
| --- | --- |
| `official_scores.reward` | Tau2 官方奖励 |
| `custom_reward.strict_success` | 启用 Agent Judge 时的严格成功判定 |
| `custom_reward.train_reward` | GRPO 用的自定义标量奖励 |

`pass^k` 对每个任务计算 `comb(成功次数, k) / comb(样本数, k)`，再按任务等权平均。每任务 4 条时，`pass^4` 要求四条全部成功。评估报告保存具体模型、任务、提示词、seed、运行配置和截断口径。

## 获取仓库

JSONL 使用 Git LFS 保存：

~~~bash
git lfs install
git clone https://github.com/3361119449-bit/agentic-rl.git
cd agentic-rl
git lfs pull
~~~

SFT 和 RL 使用独立的 veRL 环境，依赖与命令见 [RL 使用说明](agentic_rl/README.md#autodl-安装)。本仓库不对上游数据集或代码另行授予许可；使用时请遵守原始来源的条款。

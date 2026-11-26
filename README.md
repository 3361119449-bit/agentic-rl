# agentic-rl

在 Tau2 Airline 上训练和评估 Qwen3-4B 航空客服智能体。**当前实验顺序：冻结 SFT v4 → DPO → 按任务缩小 Judge 的评分范围 → GRPO。** SFT 和两条早期 GRPO 对照已有结果，接下来先完成 DPO。

## 已经完成的实验

| 阶段 | 结果 |
|---|---|
| SFT v4 | 119 个源对话、1,374 个目标轮；实际训练 117 个对话／1,338 行，冻结 step 334。首次官方测试按 30 助手轮上限，34/80 成功，pass¹ 42.5%、pass⁴ 20%。 |
| GRPO | 完成 12 步探索、官方奖励 120 步对照、自定义 v13.2 奖励 20 步。自定义 step 20 在 35 轮上限的三批测评分别成功 34、35、24 / 80。另一次仅限 24K 上下文，35 条官方通过轨迹中至少 13 条仍有严重过程错误。 |
| 历史 Judge 双审 | 自定义 20 步实际入训 638 条、9,125 个助手轮。补入工具目录后确认 599 处错标、13 处待定，另有 540 处标签正确但理由错误。已对旧路由候选做离线覆盖复核。 |
| DPO | 旧 AdamW 27 对已训练和测评。新 Muon 155 对已做 CPU 和数据验收；156 对修订包另存，待选定数据版本、做 GPU 验收并训练。 |

[实验总账](EXPERIMENTS.md)按评分协议列出结果；[官方奖励 120 步归档](agentic_rl/docs/experiments/2026-11-19-official-grpo120.md)记录训练里程碑。

## 当前实验主线

先从 SFT step 334 完成 DPO 并评估。随后设计新版 Judge：默认只检查本题终局语义；仅指定任务额外判断**当前助手轮是否正确拒绝**，逐轮 policy 违规评分留待后续。终局模板只带完成条件和必要证据，不需要轮 ID。启用逐轮评分时只给当前待评目标 ID，历史可见事实可用于判断，早先的轮 ID 不可引用。验收后从选定的 DPO 模型开展 GRPO，记录判分质量和实际 Judge 开销。

[当前状态](PROJECT_STATE.md) · [DPO、Judge 与 GRPO 交接](docs/handoffs/2026-11-26-dpo-judge-grpo.md) · [历史 v13.2 GRPO 配置与复现](agentic_rl/docs/current_experiment.md)

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

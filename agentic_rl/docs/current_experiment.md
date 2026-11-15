# 当前实验：SFT v4 后的 30 任务 GRPO

本页记录 2026-11-16 修正的正式配置。此前 smoke 已启动采样，因配置与要求不符而停止，未完成 optimizer 更新。当前只做离线修复与验证，不自动重启训练。历史流程见 [归档](../../docs/legacy-experiments.md)。

## 流程与配置

审核后的 SFT v4 → Qwen3-4B 下一助手轮 SFT → 冻结 Step 334 bf16 合并模型 → 修复任务/评分验收 → 30 任务 GRPO → 冻结官方评估。

| 项目 | 当前设置 |
| --- | --- |
| 正式配置 | `configs/rl/airline_procredit_v4_train30.yaml` |
| 任务 | 官方 train 全部 30 个；旧 6 个 internal dev 编入训练；official test 20 个保持隔离 |
| 内部验证 | `val_before_train=false`、`test_freq=-1` |
| 奖励 | `v13.1-procredit-immediate-independent`；进度权重 0.5；公式见 [奖励说明](reward_v13.md) |
| Judge | `trajectory_once_v1`：一次提交完整轨迹，返回整轨结论与全部 `simple-turn-v2` 轮次标签；高思考，新缓存，有界失败重试 |
| 用户筛选 | 关闭用户模拟合规筛选；保留对话所需的用户模拟器，不调用用户合规 Judge |
| RL LoRA | rank 32、alpha 64、all-linear；学习率 `5e-6`；PPO epochs 2 |
| 采样 | 每组 8 条，8 并发；初始 4 组，最多 3 波补组 |
| 精度 | 模型、参数与 rollout 为 bf16；reduce/buffer 为 fp32 |
| 上下文与轮次 | 16,384 tokens；35 个助手轮次上限；不设独立单轮生成上限或固定预留，工具回执完整保留 |
| 截断 | 记录原因，整条训练信号归零；训练每槽最多重采 2 次，测评不重采 |
| 历史入口 | 显式 `--stage internal_dev` / `smoke` 保留旧配置；默认入口为修复版 `full_train` |

正常评分只发一次 Judge 请求；命中缓存不发请求，格式或服务失败才按现有上限重试。不再按助手轮次单独调用 API，缺少批量结果时也不回退到逐轮调用。Policy 只发送一份，共享的可见消息通过索引引用；逐轮证据只能来自当前及此前消息，当前工具调用自己的回执不可引用。代码检查编号、覆盖、引用范围和标签矛盾，但离线检查不能证明模型的全部语义判断正确。

正式配置仍绑定 `e490e085…` 多工具系统提示词，已完成的 SFT v4 使用 `54c8a815…` 单工具版本；以配置、内容哈希和 run manifest 区分。

## 五项任务修复

| 任务 | 修复范围 |
| --- | --- |
| 7 | 统计通话开始时其他未来未取消订单的原付款，排除正在处理及过去行程，正确总额为 $708 |
| 14 | 新订单付款流水顺序不影响 DB 比较；付款来源、金额及业务状态仍须一致 |
| 21 | 使用余额足够支付的礼品卡中余额最小的一张 |
| 23 | 新订单生成 ID 和等价执行顺序不影响比较；乘客、航段及付款对应关系仍参与校验 |
| 39 | 计划变更场景只要求取消 `8C8K4E`、`LU15PA`，不要求错误取消 `MSJ4OA` |

修复定义用于用户模拟、整轨迹 Judge、终局 DB/COMM 与进度计算，不改上游原始任务文件。Task 39 必要动作、Task 7/39 语义标注同步修复。详见 [任务修复与回归](../task_repairs/README.md)。

## 启动与验收

在已安装固定 Tau2/veRL 的 RL 环境中加载自己的 `.env`，设置 `MERGED_SFT_MODEL`、`TAU2_ROOT`、`VERL_ROOT` 后：

```bash
cd agentic_rl
python scripts/prepare_tau2_dataset.py
python scripts/train_airline_grpo.py --stage full_train --reward-judge --dry-run
# 需要实际训练时去掉 --dry-run，并设置独立 --run-name 及所需更新步数。
```

当前服务器入口为 `/root/autodl-tmp/run_rl_train30.sh`，代码在 `/root/autodl-tmp/agentic-rl-train30-20261115/agentic_rl`，复用模型 `/root/sft_eval_v4_20261106_merged`。这些是该服务器的部署路径，其他机器按环境变量配置。

此前 63 项离线检查没有覆盖真实的单轮生成预算、完整回执和 Judge 请求次数，不能据此宣称正式配置已完全就绪。修正后新增生产循环行为测试：超过 1,024 tokens 的回复可正常结束；回执不裁剪；仅总上下文耗尽时停止；live 与冻结重试均使用整轨一次请求；用户筛选关闭。测试采用模拟接口和本地 tokenizer，不调用评分 API。GPU 完整单步训练仍需另行验收。

训练后的官方对照仍为 20 个 test 任务 × 4 = 80 条，关闭用户筛选与 Agent 奖励 Judge、保留官方评分；按同一提示词、35 轮与截断口径对比冻结 SFT/RL 模型。历史 30 轮基线与旧奖励的分数按原 manifest 保留，不混为同条件结果。

# 官方目标完成后回退局部信用 — 2026-10-18

基于 `0c8ee42`，新增默认关闭的 `post_completion_regression_v1`。
按用户最新要求，本次只做本地测试及离线审计；真实 PPO smoke 由用户后续运行。

## 启用及语义

修改现有 v4 配置中的字段，用原启动器的 `--config` 选择该 YAML：

```yaml
credit:
  version: procredit-turn-v4
  db_subgoal_credit: null  # 可独立设为 db_subgoal_local_v1
  post_completion_credit: post_completion_regression_v1
```

默认两项均为 `null`，关闭后保留旧 credit 审计结构及精确结果。新信用条件使用新
run，不绕过原配置/源代码的 resume 身份检查。学习率、PPO/GRPO、采样预算不变。

完成判定检查 `reward_basis` 中所有适用官方条件；当前仅支持 DB 和 COMMUNICATE，
使用原 `progress_trace.check_values`，包含最初已满足而不计入 phi 的检查。
从不以 `phi == 1` 或 Required Action 达成为官方成功的替代证明。
无 COMMUNICATE 条目时沿用官方空条件为真的语义。有 ACTION、ENV_ASSERTION、
NL_ASSERTION 等无法在现有前缀可靠验证的 reward basis，或存在 user tools、转人工
条件时，跳过该任务的新机制。仅在 `nl_assertions` 字段出现而未纳入官方 reward
basis 的条目，不是 `EvaluationType.ALL` 的适用成功条件。

仅在观测到完整完成后的官方 DB `True → False`，并有本轮 Assistant 工具的真实
`db_effect=True` 写入证据时确认回退。工具结果、顺序与轮次保持原样。同轮多个
调用共同破坏状态时限制整个 Turn；只在两次前缀之间短暂出现、没有被前缀验证
观测到的完成状态不发奖。外部失败、无法确认效果的调用、正常查询不增加惩罚。

只在存在确认回退的轨迹上，为首次完整完成 Turn 增加一次有限正信用；同轮有
policy 违规或错误/目标外写操作不给正奖。初始已完整满足目标的轨迹不发完成奖，
但随后确有写入破坏目标的轮次仍受负向限制。恢复后再次完成不重复发奖；如果
最终恢复成功，官方成功定义照旧。

正奖预算：保留原 DB 子目标信用，新增完成奖励最多为
`max(0, 0.5 − 已发放的 DB 子目标正奖总额)`。计算的是正奖总额，不用净奖励，
不能通过回退扣分重新腾出奖励预算。两个机制合计正向局部信用每条轨迹最多 0.5。

按用户明确要求，安全的首次完成轮还须保证**最终 Token Advantage 严格为正**。
任务、轨迹、局部奖励和 process 扣分全部合并后，仅对该轮执行一次：

```text
completion_floor = max(1e-6, 0.5 - process_cost_this_turn)
final_token_advantage = max(existing_token_advantage, completion_floor)
```

保底使用最大值，不再加一份完成奖励；即使 DB 子目标已用完正奖预算，仍执行此
最终符号约束。局部奖励的 0.5 上限与 Advantage 修正量是两个量：负的组优势可能
需要超过 0.5 的修正才能变正。`completion_turn_floor_adjustments` 单独记录这一修正。
process cost 会降低保底幅度；若 cost ≥ 0.5，最终正号要求优先，下限为 1e-6。
已有较大正 Advantage 保持原合并值。保底仍只属于首次安全完成轮，不奖励修复轮，
初始满足、policy 违规或混有错误写入的完成轮都不应用保底。

回退轮在任务、轨迹和原局部项合并后执行：

```text
final_token_advantage = min(existing_token_advantage, -1.0 - process_cost_this_turn)
```

这是最终值限制，不再额外扣一份 −1。同轮已有 policy 限制时取更严的结果，既不
稀释已有负值，也不重复累加同一个 policy 大小的错误惩罚。其他 Turn 的信用不受
新机制影响。policy 与回退负向限制在正向保底之后执行，违规负号优先；观察/padding
仍为 0。完成 trace 的 Turn ID 为 1-based；内部原 policy
和 Token 映射继续为 0-based。

`train_reward`、最终 `terminal_success`、官方 reward 与 Pass1/Pass4 全部不变。
中途成功过但最终失败的任务仍算官方失败。局部 trace 分别保存在 `completion_trace`
与原 `subgoal_trace`；在线评分、冻结重试和记录重建共用实现并校验冻结证据。

## 离线结果

原始数据仍为 `967e3b5` 训练归档中的 **738 条 PPO 轨迹 / 96 个原组**，逐组比较
原版、仅 DB 子目标、仅本机制、两个机制同时启用四种条件。保留成员、policy version、
旧 token/logprob 和原评分，不重新调用 Judge，不改写原记录。

| 指标 | 数值 |
|---|---:|
| 确认的完成后 DB 回退 | 56 次 / 56 条轨迹 |
| 其中初始已完整满足目标 | 52 条（不发完成奖励） |
| 训练期间首次完成后被破坏 | 4 条 |
| 获得完成正奖及最终正 Advantage 保底 | 3 条 |
| 首次完成轮已有 policy 违规而不发奖 | 1 条 |
| 原来已有 policy 负向限制的回退轮 | 56 轮（不重复扣罚） |
| 独立固定 Tau2 回放核对的 DB 前缀 | 689 个 |
| 关闭新增机制时 Token Advantage 最大绝对误差 | 0 |
| 两个机制重复/超额正奖 | 0，清单为空 |
| 新机制误处罚其他 Turn | 0，清单为空 |

每个候选轨迹的预测 DB、完整 gold DB、工具返回、DB effect 均用固定 Tau2
`a2c024725189473d2d7cea3a5cfdbcc67478e41f` 的实际工具函数体和 Pydantic 模型核对；
COMMUNICATE 前缀也使用固定源码的原谓词复核。这是依赖精简的固定函数回放，
不是完整 orchestrator/GPU 训练。回放会跳过 actor 在本地拒绝而未执行的工具轮，
包括 mixed `done` 产生的 `multiple_tool_calls` 错误。

### Task 12：`5e8df95f6e75493cbe92a20afb65c714`

T6 达成全部适用官方目标，T10 修改航班后破坏官方 DB 状态。
新增机制与是否启用 DB 子目标机制的结果相同，因为 Task 12 是单目标任务：

| Assistant Turn | 原 Token Advantage | 启用后 | 变化 |
|---|---:|---:|---:|
| T6 首次完成 | −1.7087898755 | +0.5 | 局部信用及最终正向保底 |
| T10 破坏 | −3.2087898755 | −3.2087898755 | 已严于 −1，保留原惩罚 |
| 其他正常 Turn | 原值 | 原值 | 0 |

`train_reward=0`、官方 reward=0、最终任务失败均不变。

T6 的 process cost 为 0，新增局部信用为 +0.5，抵消剩余负组优势的保底修正为
+1.7087898755，合计 Advantage 变化 +2.2087898755。这些修正不改变 trajectory
score 或组均值。三个符合条件的首次完成轮最终均为正；逐轮值及修正量见机器报告。

## 复现

```bash
PYTHONPATH=src python scripts/audit_post_completion.py \
  /path/to/original/run /path/to/new/audit \
  --tau2-source /path/to/pinned/tau2-bench
```

输出必须是新的独立目录，不能已经存在。`summary.json` 及 `groups/*.json` 保存四种条件下的逐轮
旧/新 Token Advantage、完整局部 trace、源审计指纹、Token Advantage 哈希以及
双奖/误处罚清单。任一底线校验失败则退出非零状态。聚合报告保存在
`data/audits/post_completion_regression_v1_0c8ee42.json`。

开发验证：**886 passed / 30 skipped**；Ruff 和 `git diff --check` 通过。
定向测试覆盖固定 Tau2 DB/COMM 回放、冻结重试、恢复/重复完成、同轮多工具、
初始满足、外部失败、共享预算、policy 去重、负组优势、较大 process cost 下的
最终正向保底、保底不重复相加、缺失完成轮 Token、最终负向限制与真实 Torch padding。
独立审查发现的拒绝调用回放和不一致官方/DB 状态校验问题已修复，无剩余问题。

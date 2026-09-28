# ProCredit v2：终局 hard gate 与违规轮次 credit

日期：2026-09-28。承接用户对 Judge transfer contract 和全零训练信号的修正要求。
旧设计见 `2026-09-27-tau2-procredit-reward-design.md`；本文件只定义 v2 的差异。

## 事实与边界

用户提供 `m2_procredit_smoke_seed42_v4_debug_bundle.tar.gz`。按包内轨迹记录统计，
共有 93 条，32 条标记 Judge 阶段失败、1 条 user-simulator Judge 失败；42 条有完整
reward，其中 41 条为零，5 条有非零 Phi。日志里的重复异常文本不能作为独立失败数。
这些是归档快照的统计，不能据此推断所有 refill 次数或整次作业的最终状态。

包内只有一个完整的 8 条组审计，八条终局分数、所有轮次 Phi、原始 advantage
全部为零。八条都有可由实际工具事件定位的无效 transfer，另有旧 Judge 无法可靠
定位的 policy 失败。保持原始成员和交互不变的 v2 离线诊断得到：终局分数仍全零，
`has_signal=false -> true`，1570 个策略 token 获得负 advantage，零个获得正 advantage。
这不证明前面所有步骤正确，也不是已完成 GPU optimizer 更新或 smoke 的证据。
原始轨迹及诊断产物只保存在 ignored `agentic_rl/outputs/policy_smoke_diagnostics/`。

## Judge contract

是否执行 transfer 由成功的 `transfer_to_human_agents` 工具事件确定。无成功执行时，
规范化为 `valid=false`，`applicable=task_requires_transfer`，清空不适用的 transfer
证据；不再因为模型猜测执行事实而丢弃整条 rollout。原始响应完整保留。
实际执行时 `applicable=true`；模型原本报 N/A 不构成有效性证明，不提升成有效 transfer。
判定理由、policy/semantic criterion 集合及所有仍适用的证据校验继续执行。
Prompt v8、scorer v4 更新缓存身份；旧 rubric 不能伪装成新版重用。

## 违规定位

`policy_credit` 保存零起始 `violating_turns`、归因是否完整、未定位的 rule IDs、
来源证据和冻结输入指纹。确定性安全检查使用事件 ID；无效 transfer 使用真正执行
的 transfer 事件。多个检查命中同一个 turn 只扣一次。

Judge 新字段 `violation_assistant_turn_ids` 是一开始计数的 actor 生成轮次。
已送达的 assistant 消息由冻结的 prefix 边界附上 `assistant_turn_id`；工具事件
沿用 `turn_id`。未送达输出和清理消息不补成成功进度。旧 `evidence_turn_ids`
可以指向用户/工具上下文，命名空间与 actor turn 不相同，不能猜作犯错轮次。
无法定位的遗漏不强行责罚最后一步。

存在未定位 policy 失败时，该轨迹的进度 return 置零且不参与中心化，已定位的违规
仍有局部负信号。这里的保守回退可能让某些组仍无信号；有界补采上限保持有效。

## v2 算法

终局严格成功 R、门控 V、进度 Phi、截断乘数 m、过程惩罚 P 都沿用原有定义：

`S = V * max(0, m * (R + 0.5 * Phi_T) - P)`。

轨迹 advantage 仍是组内 S 的 population-std 标准化。对归因完整的轨迹：

1. `delta_t = Phi_(t+1) - Phi_t`；已定位违规 turn 只删除正 delta，保留负回退。
2. 非违规 turn：`G_t = m * (R + 0.5 * sum(delta[t:]))`；违规 turn：`G_t = 0`。
3. 以组内归因完整且有策略 token 的所有 turn 的 G 均值中心化，包含已定位违规的
   零 G。每轮等权，不按 token 长度改变中心。未定位轨迹的逐轮 advantage 为零。
4. 已定位违规 turn 再取 `min(centered_advantage, 0) - 1.0`。扣分放在中心化之后，
   防止整个组同样犯错时惩罚又被均值消去。它不传回整条轨迹。
5. 策略 token 使用 `A_trajectory + lambda_turn * A_turn`；观测/padding 为零。
   `lambda_turn=0` 显式关闭整个逐轮项，包含违规惩罚。

这允许已验证的前期进度获得正信号，但不承诺每个早期 turn 都为正：组内中心化、
之后的负回退和轨迹 advantage 仍有影响。Phi 全零时不虚构正进度。
v2 改变训练目标；保留终局 hard gate 不等于对训练后的 policy 合规性作保证。

## 接入、审计和迁移

- 配置 `agentic_rl/configs/rl/airline_procredit_v2.yaml`，credit 版本
  `procredit-turn-v2`，reward 标签 `v6-procredit-policy-local`。
- v1 算法和原组审计的四字段 config 序列化保持原样；两版本不静默互转。
- actor 保存 credit 配置；冻结重评分重建相同归因。queue、trainer 和离线审计
  共用计算函数。`credit_from_record` 从 Judge/工具事件重建并校验归因。
- 新指标 `policy_violation_turns`、`unresolved_trajectories`；兼容名称
  `valid_turns` 在 v2 表示参与中心化的轮数，包含 gate 未通过但归因完整的轨迹。
- 保留服务器归档中的 YAML 显式注册方式，移除冗余 `@register` 导入和副作用；
  `configs/rl/agent_loop_v1.yaml` 继续提供完整 `_target_`。
- 使用新 run name。代码、prompt、reward/credit 身份变化不允许绕过旧 checkpoint
  的精确恢复检查。旧数据可离线分析，不能注入当前策略更新。

## 验证范围

覆盖 Judge 矛盾 transfer 的规范化及缓存、明确/歧义归因、违规创造进度的屏蔽、
全零组负信号、lambda 消融、生产 actor/冻结重评分/组筛选/训练张量及不可变审计。
本地真实 Torch/TensorDict 测试不能替代固定 veRL/Tau2 全依赖、真实 API 和 GPU
验收；实际 smoke 应检查 Judge 失败量、归因未定位比例、全零组比例、补采量及
至少一次 optimizer 更新。不能以“过滤器保留了组”直接声称 PPO 已产生有效参数更新。

最终本地验证：520 passed、25 skipped（真实 Torch/TensorDict）；Ruff 与
`git diff --check` 通过。独立审查发现并修复了“环境部分送达后报错导致轮次标记
提前抛错、未保存审计”的回归：已有基础设施错误时跳过 credit 标记，正常清理，
保留完整已送达会话、原始策略 token 和错误来源，不调用 Judge 或生成奖励。
该失败路径已完成先失败后通过的回归验证及独立复核。

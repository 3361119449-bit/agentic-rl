# ProCredit v4：逐轮政策惩罚与单槽位恢复

使用 `configs/rl/airline_procredit_v4.yaml` 启动新实验。v4 的训练语义取代
v3 的整轨迹 policy/safety 门控；旧配置仍保留旧算法。不同版本的 optimizer、
dataloader 和 rollout 队列不能混作一次精确续训。

## 奖励与训练信号

```text
S = m × (R_task + 0.5 × Phi_final)
policy_turn_reward[t] = -1（该 turn 违规），否则 0
A_base[t] = A_trajectory + centered_task_progress_return[t]
A_final[t] = min(A_base[t] - 1 - P_process[t], -1 - P_process[t])（该 turn 违规）
             A_base[t] - P_process[t]                         （其他 turn）
```

`S` 不再乘 policy/safety gate。R_task 是任务完成度，Phi 是去重后的进度；
截断乘数 m=0.75，否则为 1。process 不扣轨迹标量，也不改变 task/progress
return 或组内中心，只在对应 turn 最后减去固定扣分。
R_task 按任务要求选择完成分项：只有 required-transfer 任务使用转人工分支；
普通任务即使错误转人工，也继续使用 DB、交流、必要动作及语义检查。
任务未完成不凭 transfer_call 获得完成分，已完成则保留完成分。
transfer validity 不把真实完成分额外清零，
违规 transfer 的合规失败单独记录，并在实际 transfer turn 施加负反馈。
`policy_gate`、`task_safety_gate` 和 `strict_success` 仍记录严格的合规成功结果，
仅用于审计；`details.task_completion` 记录用于训练的完成值。

每轮剩余任务回报为 `m × (R_task + 0.5 × (Phi_final - Phi_before_turn))`，
随后组内中心化。policy 不裁剪正 progress，不清零违规 turn 的任务回报，
也不通过 valid 影响回报或中心。同轮多个 policy rule 只罚一份 -1；
process 独立相加，例如 policy -1 与 execution error -0.08 得到 -1.08。
其他 turn 保留任务与进度信号，不保证每个正确 turn 都为正，
因为 advantage 仍是相对组内基线计算。

execution error、unchanged retry、duplicate no progress 分别扣 0.08、0.06、
0.03。保留每次工具调用一种 base error，独立 retry/repetition 信号相加；
不同 turn 分别计费，不因总错误数增加而缩小。取消 trajectory 总 cap 和
over-turn 总 cap，暂不增加 per-turn cap；超过 soft turn limit 的每轮仍按
固定 0.02 扣分。纯 process error 只从原 advantage 减固定值，不把原正值先裁为零。
负值保证仅针对 policy 违规轮。

v4 的顶层 `process_penalty=0` 明确表示轨迹标量没有 process 扣分。
实际未封顶总成本在 `details.process_credit.total_cost`，逐轮值在
`turn_costs`，版本为 `process-credit-v2`。配置使用
`credit.process_credit=uncapped_local_additive` 与
`reward.penalty_placement=turn_only`；旧两个 trajectory cap 键在 v4 中被拒绝。

关键约束在合成 trajectory 与 turn advantage **之后**执行，确保高任务分数
也不能抵消违规轮的负反馈。同轮所有 actor token 的最终 advantage ≤ -1 - P_process；
工具/用户 observation 和 padding 为 0，原 rollout token 与 old log-prob 不变。
v4 禁止 `turn_coefficient=0`。一组八条都违规、标量全零时仍有负训练信号，
不会因为标量无方差而被过滤；组确实没有任何最终 advantage 信号时仍可过滤。

## 明确归因

v4 的训练任务选择模式为 `task_or_local_credit`。初始 Phi 没有未完成项时，
终局任务分或局部 policy/process 仍可能产生训练信号，因此保留所有已授权且标注完整
的训练任务，包括纯拒绝和政策推理任务。仍记录初始进度检查；实际没有最终 advantage
信号的组由采样器过滤。v3 继续使用 `verified_progress`。选择清单和内容指纹绑定
resume，v4 的独立筛选产物放在 `procredit_v4/`；此次模式变化须启动新 run。

真实工具事件按原 actor turn 定位，Judge 只能用显式
`violation_assistant_turn_ids` 定位文本决策。上下文证据编号不能充当违规轮。
遗漏必要 transfer 时，只接受 Judge 对具体拒绝或提前结束决策的明确归因；
不会自动惩罚任意末轮。每个违规轮必须有原始可训练 token。

缺少归因的 Judge verdict 在写入缓存之前被拒绝并有界重试；仍无法取得完整
归因时停止该槽位，保留冻结输入，不把政策失败冒充正常奖励或重新生成到合规。
新增归因要求进入 Judge 消息、缓存身份及 rubric 指纹，旧 verdict 不能静默复用。
记录中保存 `policy_credit`、`policy_turn_rewards` 以及最终 token advantage。

## 只恢复异常槽位

训练侧在原 AgentLoop 调用内部恢复，每组原有 uid/session、任务、策略版本、
采样参数和并发 lease 不变。7 条成功加 1 条交互异常时，只补采异常的那一条；
正常成员的 token 和 old log-prob 不改。policy 违规本身不触发补采。

| 情况 | v4 处理 |
| --- | --- |
| 模型生成、环境交互等明确可恢复的服务异常 | 保存失败记录，仅用新 trajectory ID 重跑原槽位，保留环境 seed |
| User Simulator 被判无效 | 沿用独立 user replacement 预算，仅重跑该槽位并更换 user seed |
| 已完成交互，但 Agent Judge/奖励评分异常 | 保留同一 ID、消息、事件、token、seed，重试冻结评分 |
| User Simulator Judge 调用失败 | 在同一冻结交互上重试筛查 |
| token/log-prob/策略版本对齐、工具结果数量、提示初始化或确定性配置错误 | 保存审计后立即停止训练，不补采 |
| 明确临时异常或 invalid user 补采耗尽 | 排空同组槽位后，保留成功评分成员；耗尽槽位的任何输出均排除 |
| 未分类异常、代码 bug 或无法确认成员终态的 worker 故障 | 立即停止训练，不按临时错误补采 |
| 任务取消 | 立即停止后续补采，等已发出的后台交互结束后释放 lease 并传播取消 |

默认 `slot_recovery.max_resamples=2`、`max_scoring_retries=2`，耗尽策略固定
为 `salvage_group`。API client 自身的重试预算仍独立存在。轨迹和输出
`slot_recovery` 元数据保存父 ID、attempt、失败 ID/阶段及两种 replacement 计数；
冻结奖励重试另存 `scoring_retries`。worker 在所有 session 结束后发布
`rollout_n`、`failed_session_ids` 和 `failure_kind`；采样器只接受明确的
`transient_exhausted` 成员证据。缺少成功槽位的输出属于 contract 错误，立即
停止，不能凭轨迹数量猜测耗尽槽位。失败 session 即使在报错前写出输出也不训练。
`group_audits/exhausted/` 记录耗尽信息；用于训练的完整或部分组另存
`procredit-group-v2` 审计，绑定实际成员与失败槽位，可精确离线重放。

7 条成功加 1 条耗尽时使用这 7 条，组均值、标准差及 turn 中心只用真实
成员。只剩 1 条时 trajectory scalar advantage 为 0；真实局部进度、policy
或 process 信号仍可训练，确实无信号则过滤。不会跨 uid 合组或伪造缺失成员。
每个逻辑批排空后使用所有有效组，不再只取四组、丢弃多余组。不足目标
四组时按 8 提示一批继续采样，最多三批；到上限后使用剩余有效组，只有
一个有效组也可更新，只有完全没有信号才跳过。失败组本身不产生额外预算。
PPO 保持 32 行 mini-batch 和两个 epoch；合成 EOS padding 只用于 batch
对齐，response/loss mask 和 advantage 均为 0，不参与组基线、审计成员或梯度。
全部轨迹使用原 token 与 vLLM old log-prob，并在更新前验证 prompt 策略版本。
本次不提供整个训练进程重启后恢复未完成组的机制。

`max_rollouts_per_optimizer_step=192` 限制的是**逻辑槽位数**。
`training/dynamic_sampling/generated_logical_rollouts` 明确记录这个口径；
旧 `generated_rollouts` 保留为同值兼容别名。
`physical_rollout_attempts` 则在每次真实交互发起前，由 Ray 作业共享计数器
累加，包括随后失败、user-invalid、无训练信号或被丢弃的轨迹。
其组成分别为 `initial_rollout_attempts`、`infrastructure_replacements`、
`user_replacements`。每个采样步骤在首批提交前取基线，避免漏掉首批或
混入上一轮费用；计数不依赖最终保留的训练 batch。
单槽位补采不占新槽位，但增加实际交互/API 成本。
`selected_groups`、`selected_real_rollouts` 记录实际进入更新的组和真实轨迹数；
`salvaged_groups`、`salvaged_rollouts` 记录耗尽组中利用的成员，
`cap_remainder_used` 记录上限处使用不足目标的余组，`rollout_use_rate` 是
真实训练轨迹数/已提交逻辑槽位数。`procredit/padding_rows` 单独记录对齐行。
默认每槽位最多 1 + 2 次 infrastructure replacement + 2 次 user replacement，
默认逻辑预算下最多可发起 192 × 5 = 960 次交互；192 不是物理尝试上限。
冻结评分重试和 API client 内部 HTTP 重试不算新的物理 rollout。
训练内验证采用相同的有界 slot 恢复；耗尽的 user-invalid 和明确临时异常不停止
整个训练，成功 sibling 继续进入验证指标。验证 sampler 不补采整个 group，
按 worker 的终态成员清单排除失败 session，即使它已提前写出部分输出。
`val/coverage/requested_slots`、`completed_slots`、`failed_slots`、
`failed_groups` 和 `completion_rate` 记录覆盖率；完成奖励指标只使用真实成功
session，全部失败时不生成奖励指标，以 `val/coverage/empty=1` 标明并继续训练。
独立 `evaluate_airline.py` 的替换仍交给原评估 driver，避免重复替换评估样本。
训练和评估均检查异常 cause/context 链及 ExceptionGroup；只认可明确的
连接、超时错误和 HTTP 408/429/500/502/503/504。未知 RuntimeError、
缺失异常证据、属性错误、模型文件不存在等默认不补采。
HTTPX/HTTPCore/AnyIO 超时的内部取消保留可重试语义，顶层任务取消仍不补采。
Tau2 后台线程在设置终止事件前保留原始异常，reset、文本步和工具步均传递
原始类型及 cause/context 链，避免把用户 API 超时误判为未知致命错误。
Judge 的三个检查数组按 criterion ID 对齐并恢复 rubric 顺序；只接受完全相同的
唯一 ID 集合，继续严格校验证据、applicable 和违规轮归因。
包装过的配置、
类型、键、断言或对齐错误也不能触发补采。失败记录保存
`interaction_retryable`；评估遇到不可重试记录或已知对齐失败阶段会停止，
不会把它当成缺失槽位重新生成。冻结评分重试仍使用原轨迹。

## 启动与兼容性

沿用 v3 的 verified-progress 训练任务筛选、canonical matcher、DB/REQ 去重、
逐次调用行李状态检查，以及显式 BF16 load/参数/rollout、FP32 规约/buffer。
筛选数据仍使用共享的 `data/parquet/procredit_v3/` 内容寻址目录。

服务器准备好固定 Tau2/veRL checkout、模型和现有 API 环境变量后，从
`agentic_rl` 目录运行：

```bash
python scripts/train_airline_grpo.py \
  --config configs/rl/airline_procredit_v4.yaml \
  --stage smoke --seed 42 --run-name procredit_v4_task_target_smoke_seed42 \
  --extra trainer.total_training_steps=1 --dry-run
```

dry-run 会运行初始 evaluator 和生成筛选文件，但不加载模型或调用收费 API；
去掉 `--dry-run` 才会启动新 smoke。v4 的 reward version 是
`v9-procredit-task-target`，credit version 是 `procredit-turn-v4`。
离线重评分会冻结新评分配置与指纹，源轨迹不改。v4 group audit 不能套用
v3 算法静默重放。旧 scorer 的精确复现应使用对应旧 commit。

## 验证边界

多工具执行已取消 schema 层的整批阻断。一个调用参数无效或名称未知时，
只生成该调用的错误结果，合法 sibling 按原顺序继续执行；原 call ID、
assistant turn 和 token/log-prob 保持不变。混合批次的错误结果通过原生
Tau2 工具消息进入完整轨迹；进度和必需动作只计算实际成功调用。
全部调用都无效时仍只返回本地错误，不推进环境。输出截断、无法完整解析
的整轮或文本混合工具调用仍拒绝执行，不能把不完整生成当作真实动作。
旧记录中的 `execution_status=blocked_by_invalid_sibling` 继续按未执行读取，
不标记 schema 错误，也不累计 process 或 unchanged-retry 成本；新混合批次
不再生成这个状态。

[固定版本 Tau2 严格回放](https://github.com/sierra-research/tau2-bench/blob/a2c024725189473d2d7cea3a5cfdbcc67478e41f/src/tau2/environment/environment.py#L340)
会重新调用已知写工具并比较结果。适配器因此在原始 assistant 消息的
`raw_data.rejected_tool_calls` 中记录被 schema 拒绝的调用及原参数。
进度和最终评分为每个独立回放环境安装临时响应处理：只有调用 ID、名称、
参数和合成错误结果全部匹配的拒绝调用才返回原错误而不执行。
合法调用继续使用原工具和严格结果比对，拒绝证据不一致仍报错。
最终评分沿用固定版本 evaluator 的原算法，通过单次调用的私有 registry
视图传入这些回放环境；不修改全局 registry、原始轨迹、Judge/沟通证据或
官方奖励组合方式。此次执行行为变化须启动新 run，不能精确续训旧源码 checkpoint。
只对 environment_reset/model_generation/tau2_tool_step/tau2_text_step
服务阶段有限补采；这些阶段若由配置/类型/键/断言错误引起也立即停止，
包括多层异常包装中的错误。
使用新 run 目录，旧 v8 记录不能当作新实验的有效评分直接续训。

2026-10-06 本地完整 `tests` 套件（真实 Torch/TensorDict）：
745 passed、25 skipped；Ruff 和 `git diff --check` 通过。
独立审查发现的 required-transfer 缓存边界和 transfer 残留门控均已修复，
并通过先复现失败、再修复的回归测试；轨迹利用与本次四项修复的独立审查未发现实质问题。
新增回归覆盖 Tau2 后台异常在 reset/文本步/工具步传递、验证单槽有界恢复、
验证全部失败与部分覆盖、Judge 合法乱序及严格 ID 成员校验、v4 零进度任务选择。
多工具回归覆盖坏调用在首/中/末位置、未知工具、非对象参数、合法调用的顺序和
逐调用 DB 变化、工具执行错误后的继续执行、原 token 保留、进度与最终评分回放、
拒绝证据不一致及合法结果篡改的拒绝。独立审查使用固定版本实际回放方法，
确认被拒绝的写调用没有重执行，严格校验仍有效，原轨迹及全局 registry 不变。
新增覆盖部分组、单成员组、耗尽槽位已写出输出的排除、剩余有效组利用、
跨补采批保留全部 80 条有效轨迹、dense/nested padding 零梯度、batch 元数据
保留及部分组审计精确重放。

回归覆盖高 trajectory advantage 下负号保持、八条全违规保留训练信号、
过程错误不降低轨迹分数、不改变其他 turn、超过旧 cap 不稀释固定值、
同轮 policy/process 相加、违规 turn 保留完整 task/progress return、
真实 Torch/TensorDict 进入生产 trainer 方法且 old log-prob 不变、8 槽只重试
失败槽、冻结评分重试/耗尽、两种取消调度顺序下停止补采并保留 lease、
包装/分组异常分类、真实 HTTPX/AnyIO 超时及断流链、评估不可重试错误停止、
耗尽组保留成功成员、致命错误停止训练、逻辑预算不因坏组扩张、跨步骤物理尝试计数、
离线重新评分及 BF16 配置约束。

本地未执行真实 GPU/API smoke，未运行依赖固定 Tau2/veRL/Ray 环境的完整契约
测试。不能据本地回归宣称 smoke 已完成或 Judge 失败率已经降低。

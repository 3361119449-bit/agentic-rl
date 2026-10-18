# DB Subgoal Local Credit v1 — 2026-10-18

按用户提供的实施方案，基于 `967e3b5` 增加可选局部信用。现有标量
`train_reward`、`progress_trace.phi`、官方成功判定、学习率、采样预算和 PPO 不变。

## 开关与定义

在 `configs/rl/airline_procredit_v4.yaml` 中，默认关闭：

```yaml
credit:
  version: procredit-turn-v4
  db_subgoal_credit: null
```

启用时把该字段设为 `db_subgoal_local_v1`，用现有启动器的 `--config` 选择修改后的
配置；其他字段沿用原文件。只允许 ProCredit v4 使用此信用版本。

通用适用条件：DB 任务，同一订单至少两个不同的目标更新工具，无依赖、无其他
必要写操作、无转人工目标、无未经验证的 user tools。本批数据自动识别为任务
17、21、33；12 的单目标、14/23 的取消后重订、42 的多订单取消不启用。

初始订单直接从 live DB 只读冻结，不增加 actor 工具调用。后续按 `ToolEvent.sequence`
处理原始工具返回的完整订单快照，并同时使用原有必要动作语义匹配器。
航班必须包含完整顺序、日期和 cabin；乘客完整匹配姓名/生日，忽略列表排列；
行李必须同时匹配 total/nonfree。支付约束沿用原 matcher，包括经过真实前置
nonfree 状态验证的零费用例外。矛盾的 `state_before` 不能用来豁免支付约束。

目标状态另存为版本化 `subgoal_trace`，不替换原进度。每个独立目标首次正确完成
在自身轮次获得 `0.5/N`，初始已正确的目标不给正奖。同轮有 policy 违规、错误、
目标外或无法验证的写入时，暂缓整轮新增正奖；不能移到后续查询轮发放。
这是 Turn 粒度信用的限制，不会阻断合法工具执行。

已验证目标被后续写入破坏，在破坏轮扣 `0.5/N`，每个目标最多扣一次，累计不超过
0.5；早期正确信用保留。每个目标最多获一次正奖，重复修正不能刷分。未知快照或
DB effect 不证明发生回退。发送补偿券会阻止同轮正奖，但不抹掉订单的已验证状态。
查询、确认和合法额外调用不新增统一惩罚。

局部项加入原 Turn Advantage 后，继续扣原 policy/process cost，并保留最终
policy token 的严格负向限制。组均值、轨迹 Advantage、原 Token/logprob/mask
均不变；padding 和观察 token 始终为零。原配置省略或关闭开关时，credit 审计字典
也不新增字段，便于精确复现。

## 离线验收

原始归档：`m2_procredit_v4_967e3b5_fresh_ep2_seed42_20261013_full_analysis_bundle.tar.gz`。
SHA256：`a656ca8be508f201435bb3e16e8c4ff624e36b8fc160f44b910992e081270cb3`。

重算保留原始 Group Audit 的成员、slot、policy version 和冻结 Judge 结果；
同时核验源审计/输入指纹。每个组必须先完全复现原始关闭状态结果，才能比较新信用。
原记录、官方 reward、Judge、token 和旧 logprob 不被改写，不调用 LLM。

| 指标 | 结果 |
|---|---:|
| 原 PPO 轨迹 / 原组 | 738 / 96 |
| 满足适用规则的轨迹 | 93（17、21、33 各 31） |
| 验证到部分目标进度的失败轨迹 | 26 |
| 获得正局部信用的轮次 | 23 |
| 正确轮次从非正 Advantage 变正 | 12 |
| 原来无正 Advantage、现在有正 Advantage 的组 | 5 |
| 原来完全无信号、现在有信号的组 | 0 |
| 混合写入或 policy 原因暂缓的轮次 | 12 |
| 实际固定工具回放验证的订单写入快照 | 80 |
| 复核最终负向限制的 policy 违规轮次 | 2,914 |
| 关闭开关时 Token Advantage 最大绝对误差 | 0 |
| 错误/目标外写入因新增机制从非正变正 | 0（清单为空） |
| 回放无法验证而排除的轨迹 | 0 |

本批样本没有出现经过验证的目标回退（0 轮）；回退、重复恢复、初始满足、混合
快照缺失、错误支付、旧前置状态和 padding 等边界另外通过单元测试覆盖。
上述“5 个组”表示新增正信号，不能称为救回 5 个全零组；原组已有 policy 负信号。

### 指定轨迹的旧/新对比

轮次使用 1-based Assistant Turn。下面的数值变化仅为新增信用，不改变标量奖励。

| 轨迹前缀 | 核对结果 |
|---|---|
| `c968824f` / 17 | 识别航班、乘客两个目标；T11/T13 各增加 1/6，0 → 1/6；错误行李轮不增加信用。 |
| `d7933aad` / 17 | 同轮正确航班、乘客与错误行李混合，整轮新增信用为 0。 |
| `726dd4f3` / 33 | T17 正确行李信用 +0.25，Advantage −0.02 → 0.23；此前错误航班轮保持原值。 |
| `e74ac7f0` / 17 | 错支付的航班轮不获奖；独立正确的乘客/免费行李轮各增加 1/6，最终 Advantage 仍为负。 |
| `4c740ae4` / 33 | 早期正确轮保留 +0.25；后续目标外补偿轮没有新增正信用。 |
| `590abeb3` / 21 | 订单 ID 不能替代航线/支付检查；新增信用全部为 0。 |
| Task 42 | 不启用局部子目标；额外错误取消不会因新机制获奖。 |

复现命令（输入为解包后的原 run 目录，输出必须是独立的新目录）：

```bash
PYTHONPATH=src python scripts/audit_db_subgoals.py \
  /path/to/original/run /path/to/new/audit \
  --tau2-source /path/to/pinned/tau2-bench
```

输出 `summary.json` 及 `groups/*.json`，包含逐轮旧/新 Advantage、delta、局部 trace、
源组指纹、前后配置、Token Advantage 哈希、policy/错误写入标记和误奖励清单。
不满足底线时脚本以非零状态退出。聚合结果保存在
`data/audits/db_subgoal_local_v1_967e3b5.json`。

回放固定在 Tau2 `a2c024725189473d2d7cea3a5cfdbcc67478e41f`：使用该提交
未修改的 Airline 工具函数体、Pydantic 模型和初始 DB，验证返回值、真实当前订单、
DB effect 以及行李执行前状态。为避免导入 LLM runner，仅替代 toolkit 初始化、
DB 装载外壳和日志；这不是完整 orchestrator 集成测试。缺少初始状态或不能覆盖的副作用会禁用
相应局部信用，不猜测状态。

新条件应使用新 run，可将 Step 12 权重作为起点，不绕过原 optimizer-resume
身份检查。本次没有启动 GPU rollout 或 A/B 训练；新增信用不代表 Official Pass
已经提高。

开发验证：完整测试 **848 passed / 30 skipped**，包括真实 Torch 的局部信用与
padding 检查；Ruff、`git diff --check` 通过。独立审查发现的行李前置状态和
离线取消回放日志缺口已修复，最终审查无剩余问题。

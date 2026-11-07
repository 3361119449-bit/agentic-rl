# 简单逐轮奖励（2026-11-02）

默认训练配置 `configs/rl/airline_procredit_v4.yaml` 启用
`credit.final_turn_rule: simple-turn-v1`，奖励版本为 `v10-procredit-simple-turn`。
以原 main `08cc07a` 的完整 ProCredit v4 最终优势为 A0：

```python
if bad_turn:
    advantage = min(A0, -1.0)
elif eligible_correct_refusal:
    advantage = max(A0 + 1.0, 1.0)
else:
    advantage = A0
```

规则位于原轨迹项、局部项、过程成本、组内中心化和旧政策负向约束之后，
只修改 actor token 的最终优势。轨迹总分、组均值、old log-prob、观测及 padding
掩码保持原路径。原有更强负值不会被改成 −1；拒绝奖励不会再次加入轨迹总分。

## 标签与证据

`bad_turn` 合并原政策违规、明确工具错误、参数/事实/政策语义错误。
明确工具错误仅包括 `parse_error`、`unknown_tool`、`schema_invalid`、
`model_caused_execution_error`、`mixed_content_and_tool_call`。
同轮任一工具出错，整轮 actor token 应用规则。多工具本身、正常空结果、done、
重复查询、超软轮数和修复行为不会自动新增坏标签或专门奖励；原有软成本仍保留。

语义 Judge 按生成轮逐次读取 `record.messages` 的实际可见前缀及绑定 policy。
请求不包含隐藏任务答案、完整环境回执或未来消息。未知不等于事实相反，
但无依据地声称“已经检查/确定/成功”可判坏。必要条件不明时不强授拒绝奖。
同轮拒绝正确但附带错误事实或非法建议时，坏标签优先。

正确拒绝按“目标对象 + 被禁止操作”首次计奖。同一轮的多个拒绝分别登记请求键，
整轮最多加一次 +1；之后重复解释其中任一请求不再加奖。后来明确承诺/执行同一个仍非法请求
可撤销此前授奖资格，但不把原拒绝轮改判为坏。新事实使操作合法、其他子请求、
独立无效转人工不会自动撤销。同轮多个拒绝只撤销对应请求，若仍有其他首次有效拒绝，
该轮保留一次奖励。索引以 0 为起点；工具事件 turn_id 仍按原记录从 1 起。

每条轨迹额外执行至多与 assistant 生成轮数相同的语义评分请求（失败有界重试），
复用当前 Judge 的模型、服务地址、API key、并发预算和传输重试。
这会增加未来训练的 Judge 调用成本；缓存命中不再次请求。
缓存位于当前 Judge 缓存目录下的 `simple_turn_target_v2/`，与旧整轨迹评分隔离。

语义服务临时失败只重试同一份冻结轨迹的评分；不会因此重新生成 Agent 轨迹。
标签缺失、证据越界或输入变化会阻止该记录进入更新；不会用中性标签凑齐。
评分重试耗尽后的组成员处理继续遵循已有 v4 策略。

## 运行

在 VS Code 的远程 Linux 终端，进入本仓库 `agentic_rl` 目录并激活原训练环境。
先按现有 README 准备数据，并设置已有的 `TAU2_ROOT`、`VERL_ROOT`、
`MERGED_SFT_MODEL`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_API_KEY`、
`DEEPSEEK_USER_MODEL`、`DEEPSEEK_USER_SIM_JUDGE_MODEL`、`DEEPSEEK_JUDGE_MODEL`。
仍使用固定 Tau2/veRL 提交，不需要修改官方 veRL 源码。

```bash
# 检查配置及将执行的命令，不启动训练。
python scripts/train_airline_grpo.py \
  --config configs/rl/airline_procredit_v4.yaml \
  --stage internal_dev --epochs 2 --seed 42 \
  --run-name simple_turn_v1_seed42 --dry-run

# 实际训练：确认环境与输出目录后，由你执行。
python scripts/train_airline_grpo.py \
  --config configs/rl/airline_procredit_v4.yaml \
  --stage internal_dev --epochs 2 --seed 42 \
  --run-name simple_turn_v1_seed42
```

首轮仍为 4 组 × 8 槽位（32 条），过滤后只补缺少的组，保留原有用户筛选和
最终优势非零过滤。本次正常训练会按当前策略采集轨迹，新增的是末端奖励规则，
没有加入历史轨迹重放方案。

奖励配置已变化，请使用新的 run 名称，从你选定的合并权重启动。
旧 run 的优化器恢复仍要求配置身份完全一致；不要把旧 run 目录当新实验输出。

旧方案完整保留在 `configs/rl/airline_procredit_v4_legacy.yaml`，运行时用
`--config configs/rl/airline_procredit_v4_legacy.yaml` 和另一个 run 名称即可。
其他旧版本配置不受默认切换影响。旧轨迹没有新语义标签时，纯离线重评分脚本
会拒绝直接套用新模式；它不会调用模型或自行补造标签。

官方测评入口和 pass^1/pass^4 的定义不变。独立评估默认开启整轨迹 Agent 奖励
Judge，同时报告官方和附加 strict 指标；`--no-reward-judge` 仅保留官方指标，
独立 User Simulator 合规检查仍开启。评估 YAML 未启用 `simple-turn-v1`，
不会把上述训练 token advantage 规则当作官方评分。

## 审计与验证范围

- 轨迹：`scoring_inputs.simple_turn_inputs` 保存绑定前缀；
  `metadata.simple_turn_labels` 保存原始决定、原因、消息证据、去重键、撤销轮次与标签。
  `judge_calls` 保存每次返回、prompt hash 和 cache key。
- 训练组：`group_audits` 的 `credit.trajectories[*].simple_turn_audit`
  保存每轮 `a0`、两个布尔标签、证据、`anew`；token 仍按原 response_turn_ids 对齐。
- 本地测试覆盖报告数值样例、去重/撤销、工具错误集合、因果证据边界、缓存重试、
  生产 rollout 与冻结评分入口、配置兼容，以及实际 PyTorch 张量和 padding。

本次开发没有运行 GPU 训练、真实模型测评或付费 Judge。自动语义标签的判断质量
仍需在你的真实训练轨迹上抽查，不能凭数值测试宣称模型行为已经改善。
报告中 738 条历史轨迹的原始数据不在本次工作区，未声称重新完成其全量重放。


## 2026-11-07: explicit target binding (simple-turn-target-v2)

The final advantage rule remains `simple-turn-v1`; no scalar formula, policy
penalty, refusal deduplication/revocation, or actor-token mapping is changed.
The semantic Judge request/schema/cache version is now `simple-turn-target-v2`.
Frozen `simple_turn_inputs` and original actor messages remain unchanged.

The Judge view has a separate `current_generation` and `target_message_index`;
`history` contains only earlier causal messages. Actor/native `turn_idx` and
`assistant_turn_id` metadata are omitted from this presentation to remove the
zero-based/one-based ambiguity. Message text, raw generation and tool arguments
are preserved. Future messages, hidden task answers and undelivered full tool
returns are still excluded.

Every response, including a neutral decision, must echo the target position and
supplied SHA256, quote a literal 16-160 character substring of the current raw
text (the whole text when shorter), and cite the current message in its evidence.
Wrong identity, old-generation quote, missing current evidence, future evidence,
and incomplete schema fail validation and use the existing bounded frozen-score
retry path. A hash/quote check binds the selected object; it does not certify the
semantic truth of all policy or arithmetic judgments.

New traces contain `judge_version: simple-turn-target-v2`, and the Judge uses
`simple_turn_target_v2/` with new prompt/rubric/schema/scorer version keys. Old
cache files and target-unbound traces are preserved for audit but cannot enter
training silently. Regrade all turns from their unchanged frozen actor-visible
inputs and recompute group advantages; do not shift or relabel old decisions.
Use the real Judge when regrading; no neutral labels or quotes may be fabricated
from a legacy response just to satisfy this schema.

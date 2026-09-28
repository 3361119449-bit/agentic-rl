# ProCredit v3：奖励 contract 修复

新实验配置：`configs/rl/airline_procredit_v3.yaml`。从新 run 开始，不能把
v1/v2 的 optimizer、dataloader 或旧 rollout 队列当成 v3 恢复。

## 行为

- Completion matcher 按固定 Tau2 工具消费的 FlightInfo、Passenger、Payment
  字段比较，忽略工具会丢弃的嵌套附加字段；航段顺序、ID、金额仍须匹配。
- Safety 独立检查工具与目标 reservation/user 的写入范围；允许同一目标的中间
  修改。booking/certificate 额外限制创建次数。Judge 继续负责用户确认、
  资格与支付政策，最终完成度继续由官方 DB 与 completion 判断。
- 行李 matcher 保存每次调用前的付费行李数。只有能证明本次增量费用为零，
  才允许不同 payment_id 等价；同批次第二个工具读取第一个工具执行后的状态。
- Judge transfer 继续以实际成功调用为准，未 transfer 时规范化虚构的适用/
  有效结果。新 policy rubric 要求显式 `applicable`：N/A 必须通过、写明原因、
  不携带违规归因；真实工具调用即使失败，调用前确认等政策仍适用，
  不能被标为 N/A。退款完成、transfer 成功等事实仍须实际执行成功。
- `progress-v2` 在 DB 检查适用时只用 DB + COMM 计算 Phi，REQ 仍保留审计记录；
  没有 DB 时保留 REQ progress。初始已满足的检查不计分，DB 回退仍扣 progress。

最终标量奖励保持：

```text
S = V × max(0, m × (R + 0.5 × Phi_final) - P)
```

违规时 V=0。v3 另将过程错误映射到生成该错误的 turn：先按原规则去重和封顶，
再按比例分配，所有 turn cost 之和仍为 P（上限 0.2）。局部反馈为
`min(centered_turn_advantage, 0) - max(policy_penalty, process_cost)`，
仅作用于有该 penalty 的 turn；不向之前正确步骤传播。两类 penalty 同 turn
取较大值。trajectory advantage 的原公式保持，`turn_coefficient=0` 关闭
全部局部信号。这让同组标量全零时的真实过程错误仍可产生训练信号。

## 数据与精度

v3 启动前用固定 Tau2 evaluator 检查训练任务的真实初始状态和初始消息历史，
不调用 LLM。没有未满足 progress 检查的任务被排除；全部排除会直接报错。
这只减少已批准的 train/smoke 子集，internal-dev/test 评估数据保持完整。
不会给纯政策推理任务编造 progress，因此当前训练覆盖面会变窄。

原始 parquet 保留。新文件位于 `data/parquet/procredit_v3/<digest>.parquet`，
相邻 JSON 记录保留/排除原因与检查指纹。筛选源内容、manifest、最终 parquet、
代码与配置均进入 resume identity。启动输出列出实际任务 ID，不能仅凭历史
smoke ID 列表判断本次入选情况。

显式设置模型 load、FSDP 参数与 rollout 为 BF16；规约与 buffer 保持 FP32。
相冲突的 dtype `--extra`、父级配置覆盖或删除会在启动前被拒绝。
固定 veRL 的 FSDP YAML 没有 `mixed_precision` 字段，因此通过 Hydra
`+actor_rollout_ref.actor.fsdp_config.mixed_precision` 添加。
依据：[固定 FSDP engine YAML](https://raw.githubusercontent.com/verl-project/verl/483b8a009ba3a97563edee3a19887e4862b8094a/verl/trainer/config/engine/fsdp.yaml)、
[实际加载及混合精度实现](https://raw.githubusercontent.com/verl-project/verl/483b8a009ba3a97563edee3a19887e4862b8094a/verl/workers/engine/fsdp/transformer_impl.py)。

## 启动

在服务器准备好固定 Tau2/veRL checkout、模型和既有环境变量后，从
`agentic_rl` 目录运行：

```bash
python scripts/train_airline_grpo.py \
  --config configs/rl/airline_procredit_v3.yaml \
  --stage smoke --seed 42 --run-name procredit_v3_smoke_seed42 \
  --extra trainer.total_training_steps=1 --dry-run
```

dry-run 会运行无模型的初始 evaluator，并生成筛选 parquet/manifest；不会启动
模型或调用收费 API。检查输出后去掉 `--dry-run` 即可开始新 smoke。
v1/v2 原有纯 group-credit 审计数学仍保留，但共享 matcher/Judge contract 已升级；
复现旧 scorer 应检出对应旧 commit，不能只切换 YAML。

v3 离线修改 process 权重重算时，输出记录同步冻结新评分配置和指纹，
并保留来源评分指纹与 reward version；源轨迹文件不改动。

## 验证范围

2026-09-28 本地最终 `tests` 套件（启用真实 Torch/TensorDict）：
565 passed、25 skipped；Ruff 与 `git diff --check` 通过。
独立审查发现的失败写操作 applicability 和离线重算配置归属问题均已通过
先复现失败、再修复的回归测试。

离线回归覆盖 matcher、逐调用行李状态、Judge applicability/transfer、progress
去重、零分过程错误、actor 冻结重试、queue 筛选、Torch advantage 与
TensorDict、训练筛选文件及精度/恢复身份。

原始 debug bundle 中 task 11 的轨迹 `0f78fa48677a4a69860fcb90da5673e5`：
required-action 从 0 修正为 1，task-safety 从 false 修正为 true。该轨迹原来
另有 policy gate=false；matcher 修复不代表它已合法成功。

本地尚未执行真实 GPU/API smoke，也未运行完整 Tau2/veRL 契约测试。
当前环境缺少 Ray 和固定 Tau2/veRL checkout，SFT 契约测试还需要
`SFT_VERL_ROOT`。本地通过不能证明 smoke 已完成或 Judge 拒绝率已下降；
需在服务器的新 run 核对这些指标。

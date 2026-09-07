# 复审修复与验收边界

## 2026-09-07：pass^4、RL 恢复身份与跨版本 tokenizer 回归

本轮基于 `93d4bb8`，只修改报告、RL 恢复保护及测试/文档，不改变数据、奖励权重、
学习率、LoRA 配置或 SFT 训练算法。

- RL 报告改为官方 Tau2 pass^k 的 `comb(c,k)/comb(n,k)`，与独立 SFT 报告一致。
  4 次中仅成功 1 次时 pass^1=0.25、pass^4=0。旧版 RL 的 pass@4 报告需要从
  完整原始轨迹重新汇总；新报告显式带 `metric_definition` 和公式，原有字段名保留。
- RL 启动保存 `rl_resume_identity.json`，真实 trainer 保存 checkpoint 时一并快照。
  训练阶段、train/dev Parquet 内容、标注/AgentLoop 配置内容、最终命令中的种子、
  epochs、总训练步数及其他训练 overrides 都被绑定。换到空目录也核对原 checkpoint，
  不能借此修改学习率或训练数据。拒绝恢复发生在 GPU 子进程和首次输出写入之前。
- 缺少 checkpoint 自身身份记录的旧版/不完整 checkpoint 拒绝直接恢复，不从今天的
  配置补造身份。原模型文件保留；可导出/合并权重开启新实验，但不是精确恢复。
- 真实 Qwen 测试的参考编码显式使用固定 veRL 包装器的 `return_dict=False`，兼容
  Transformers 4/5；没有因此修改已正确指定该参数的生产 prompt 编码器。
  CI 增加 4.57.1 / 5.10.4 双版本矩阵，并在真实 RL 依赖解析环境中运行 tokenizer 测试，
  防止“CPU 旧版本通过、实际 RL 新版本未覆盖”。

回归覆盖公式边界与 SFT/RL 一致性、实际 launcher 的首次启动/同配置恢复/跨目录恢复、
变更 CLI 或同路径数据后拒绝、拒绝路径不改写旧 run、旧 checkpoint 无清单拒绝，
以及实际 trainer 保存方法中的身份快照。本地完整 CPU 回归在 Transformers 4.57.1
和 5.10.4 下各 **200 项通过**；LoRA SFT 隔离检查 **8 项通过、3 项缺少完整依赖跳过**；
独立 Tau2 rollout/SFT 流程 **3 项通过、1 项未启用 Hub tokenizer 集成而跳过**，lint 通过。
完整 Linux 依赖层的最终结果见本轮提交后的 Actions 记录；GPU/API、真实梯度更新、
权重同步、BF16 合并等价与 GPU 断点恢复仍未验收。

以下为历史复审记录；旧 pass@k 和旧恢复说明以本节及最新 README 为准。

## 2026-09-05：针对 `044f4c5` 的再审

此前“已可直接进入 GPU smoke”的结论过早：两处固定依赖启动错误确实存在，
不是 GPU 才能发现的问题。以下为本轮增量，后续章节保留上一轮历史，不代表本轮验收。

- Runner 不再继承 Ray ActorClass；普通类在入口只装饰一次，保留固定版 manager 构造契约。
- 评估显式设置 actor mini/micro-batch 和 rollout log-prob micro-batch；`val_only` 不更新权重。
- `generation_truncated` 与预算/轮次外部停止统一令官方成功为 0，保留自定义已完成状态奖励；清理不改写真实终止原因。
- 完整环境轨迹在 cleanup 前快照，Judge 和转人工沟通评分只用这一视图，不读取未送达输出或裁剪后的观察；prompt 版本升至 v4。
- 失败但改库的工具进入确认/任务安全审计，有独立失败写入检查；不算必须动作完成，一次性确认也会消费。异常抛出路径记录前后 DB hash 的变化，不偷偷回滚官方环境。
- worker 数、全局轨迹数、两类 API 请求数及 vLLM 序列数分别配置。runner 先创建共享预算 actor，跨 worker 使用同一额度，记录实际峰值；异常 worker 的额度不静默回收。
- SFT/RL 训练保存实际底座内容指纹，adapter 导出传递身份，合并与等价性验证先核对身份，拒绝移动的 Hub 默认分支或错误权重。数值等价不能代替底座一致。
- 已完成交互但 Judge/自定义评分失败的 slot 单独列为 scoring pending。保存冻结评分输入及指纹，原 ID 原地原子更新，只重评、不重新 rollout；仍不完整就拒绝最终指标。

本轮未改变学习率、LoRA rank、奖励权重、PPO epoch、数据集或任务标注。
当前本地组合回归 **105 项通过**，含生产 loop + 模拟环境的截断/观察测试，lint 通过。
代码提交 `151d850618c6c37191bd2f3c8beca2a72f579332` 的
[GitHub CI](https://github.com/3361119449-bit/agentic-rl/actions/runs/33968645059)
两层均已通过：105 项 CPU/真实 tokenizer 回归，以及 **5 项真实依赖契约测试**。
后者实际覆盖 runner 导入与单次 Ray 装饰、训练配置、基线评估配置、带 adapter
评估配置的 Hydra/veRL 校验，以及两个真实 Ray worker 的共享额度。
本机 Docker daemon 未运行且没有可用 Linux RL 栈，所以真实依赖层交由 Linux CI 执行。
首轮真实依赖 CI 确实拦截了 vLLM 0.18 的 Transformers <5 冲突；现采用固定 veRL
[官方安装脚本](https://github.com/verl-project/verl/blob/483b8a009ba3a97563edee3a19887e4862b8094a/scripts/install_vllm_sglang_mcore.sh)
指定的 vLLM 0.24.0，而非忽略包依赖。安装保留 `pip check`。
完整依赖安装随后通过，并发现第三方 `scripts` 包遮蔽本项目 launcher 的问题；本项目
`scripts` 现为显式包，修复后的真实入口测试通过。
该 CI 的依赖组合为 veRL `483b8a00`、Ray 2.58.0、vLLM 0.24.0、
Transformers 5.10.4、torch 2.11.0、TransferQueue 0.1.8，`pip check` 通过。
契约测试未加载 4B 权重、未调用 DeepSeek；日志中的可选非 FSDP 引擎缺失和弃用
warning 不代表那些引擎已验收。官方数据核验仍为 30 train / 24 RL-train /
6 internal-dev / 20 test，50 个保留动作的内容和格式均与官方一致。
GPU/API、两个 PPO epoch 内部 old-log-prob、模型权重导出与 BF16 等价性仍未验收。

---

本次以 `c4883f736333461fc87e0e1c25c9d857e69b27b5` 和用户提供的复审意见为基线，重新核对固定版 veRL/Tau2 源码后修改。原始 SFT 数据、清洗后 JSONL、官方任务 ID、必须动作内容和动作依赖均未变更。

## 已修复

| 问题 | 实际改动 | 回归覆盖 |
| --- | --- | --- |
| P0：多轮分隔符遗漏 | `chat_stream.py` 在每次工具/用户 observation 前追加 `turn_separator`，并将其计入二分预算；分隔符的 mask/log-prob 都为零 | 真实 Qwen tokenizer：工具、用户、synthetic error、连续两次工具、截断边界、仅差 separator 的预算 |
| P0：未展示 proposal 可被确认 | 只有成功 `step_text` 后的纯文本能注册 proposal；失败写调用不改变 pending；工具名与参数绑定，一次性消费 | 直接写→模糊询问→Yes 仍被拒绝；显式提案成功；改参数/重复执行/混合标签拒绝 |
| 同轮解析不完整 | 原始开始标签、结束标签、完整块与成功解析数量必须一致且等于 1；除终止 EOS 外不能有附加文本 | 一个合法块加损坏块、两个合法块、text+tool、proposal+tool 均整轮拒绝 |
| 相对恢复路径错误 | 相对 `agentic_rl/` 解析为绝对路径；要求 `global_step_N`、actor 模型/优化器/extra-state 分片及 `data.pt` | 路径、缺文件、计数错位均在启动前检查 |
| 恢复计数回退 | 每个 checkpoint 保存自己的 `step_counters.json`；恢复验证 optimizer step 一致；旧 run 级计数只在恰好匹配时兼容 | 较旧 checkpoint 不会读取后续训练的计数；跳过时权重版本标记不虚增 |
| 评估混样与缺任务 | 不可变 manifest、任务/样本 slot、单进程锁、显式 resume、只补基础设施失败或缺失 slot；模型失败仍是有效样本 | 真实 launcher 的模拟子进程测试：80→缺 1→只补 1；同 tag 新实验拒绝；换模型 resume 拒绝 |
| YAML 不生效 | YAML 直接生成 LoRA/lr/PPO/clip/rollout/dynamic/vLLM overrides；受支持的 CLI 覆盖反映到 runtime snapshot | 修改 YAML 确实改变最终 veRL 命令；冲突的固定约束会报错 |
| SFT revision 不一致 | 先下载指定 revision 的本地 HF snapshot，预处理 tokenizer 与 `model.path` 共用该目录 | 校验 snapshot 的 revision 参数以及真实训练路径来源 |

额外保护：模型生成若没有结束 EOS，记为 `generation_truncated`，不执行残缺动作，也不凭空补 EOS 继续拼接。原始生成文本另行保留供审计。确认采用保守的、无附加条件的 Yes 协议；包含修改或条件的答复必须重新展示 proposal，不由字符串前缀直接授权。

额外发现并修复：固定版 veRL 默认只剔除没有任何可用轨迹的失败组，可能保留“7 条成功 + 1 条 API 失败”的部分组。本项目训练采样现在剔除所有失败组，不让部分组进入要求 8 条有效轨迹的 GRPO 更新。验证分支不作此训练过滤，仍通过评估 slot 补齐机制处理失败。

## Policy 规则改变

原来每任务一个笼统 Judge 条目，现拆为 22 个可定位规则，覆盖身份获取、确认、单动作轮次、乘客、支付、行李、保险、航班/舱位修改、取消、退款、补偿和转人工。规则未触发时按不适用通过；不能把任务没完成本身当成政策违规。每个失败条目必须提供 evidence turn IDs 和具体理由。生成代码仍在 `scripts/build_annotations.py`，规则源在 `src/tau2_agentic_rl/policy_rules.py`，原有 train/test 物理隔离保留。

特别纠正复审中一个示例：**basic economy 不是一律禁止取消**。没有已飞航段且满足 24 小时、航司取消、商务舱或保险承保原因之一时可以取消；“不可改航班”和“不可取消”不是同一条规则。依据为 [固定 Tau2 Airline Policy](https://github.com/sierra-research/tau2-bench/blob/a2c024725189473d2d7cea3a5cfdbcc67478e41f/data/tau2/domains/airline/policy.md)。

Judge prompt/rubric 版本与缓存键已更新。轨迹新增 rubric 内容指纹；离线重打分拒绝 rubric 已改变或没有指纹的旧记录。不能把旧的一条总判断当成新的 22 条判断使用。离线打分必须输出到新的空目录，且不会沿用冻结评估 manifest 身份。

这不等于 Judge 已经过人工校准。正式训练前仍需审计 50–100 条代表性轨迹、逐规则统计误判率，特别检查规则不适用时是否误判。

## 评估与恢复怎么使用

以下命令均在 `agentic_rl/` 下执行，环境安装与模型导出见 README。

```bash
python scripts/evaluate_airline.py \
  --split official_test --samples 4 --tag sft_grpo_frozen_v2 \
  --model-path "$MERGED_SFT_MODEL" \
  --lora-adapter /root/models/qwen3_4b_airline_rl_export/lora_adapter \
  --tau2-root "$TAU2_ROOT" --verl-root "$VERL_ROOT"
```

相同模型和全部参数的中断重试：在同一条命令后加 `--resume`。每次启动最多初始轮加 2 轮补齐，可用 `--max-refill-rounds` 修改重试上限。每个 slot 的环境 seed 和每轮生成 seed 固定；这不能保证外部 API 或不同 GPU 调度完全确定性。

最终 test 必须有 20 个任务 × 4 条有效轨迹 = 80 条；少一个 slot 也不输出最终 pass^1/pass^4。基础设施故障会保留记录、重试缺失 slot；合法模型失败不会重试到成功。非空 tag 不能新建；同 tag 恢复时模型权重、adapter、代码、配置、标注和采样身份必须一致。旧版无 manifest 的目录不能直接混入新评估。

```bash
# 进行中只看缺失信息，不输出任何最终分数
python scripts/summarize_evaluation.py \
  outputs/evaluations/sft_grpo_frozen_v2/trajectories --allow-incomplete
```

若主机被强制关闭留下 `evaluation.lock`，先确认旧评估进程及其 Ray worker 已完全停止，再人工移走该锁并使用 `--resume`；不要在旧进程仍运行时强行解除锁。

RL 恢复的路径相对项目而不是 veRL checkout。继续相同实验须通过原 checkpoint 的
`rl_resume_identity.json` 校验；可换为空的新 run 目录，但仍必须保持训练身份。
改训练配置应导出/合并权重后开启新实验，不传 `--resume-from-path`。
`runtime_config.yaml` 固化实际配置，每次启动另存 `launches/<id>.json`，不覆盖前一次启动记录。

## PPO 审计：已接入与尚未证明的内容

```bash
python scripts/train_airline_grpo.py --stage smoke --run-name smoke_ratio_v2 \
  --tau2-root "$TAU2_ROOT" --verl-root "$VERL_ROOT" \
  --extra trainer.total_training_steps=1 \
  --extra trainer.save_freq=1 \
  --extra +trainer.ppo_audit=true
```

开启后，真实 trainer 在更新前额外调用当前 actor 的 log-prob 推理，和 vLLM rollout old log-prob 比较，不覆盖 old 值。报告保存在 run 的 `ppo_audit/step_N.json`，包含 policy-token ratio mean/std/p01/p50/p99/max deviation，以及整个更新前后的 old-log-prob SHA-256。非有限值、非 policy token 的非零 old 值和超过默认平均绝对 log-prob 差阈值 0.005 都会阻止更新。该开关会增加计算与传输开销，适合 smoke。

**重要限制：目前哈希检查发生在整个 actor update 前后，不在 worker 内部每个 PPO epoch 边界。** 报告明确标注 `epoch_boundaries_instrumented=false`。不得把两个端点相同宣称为“已实测 epoch 1 与 epoch 2 输入完全一致”。仍需在固定版 worker 上完成逐 epoch GPU 验收。并未为了获得漂亮的审计结果而将正式 PPO 的多个 epoch 拆成多次 actor RPC。

## 本次验证与未执行项目

- 本地最终组合测试 **91 项通过、0 项跳过**（含真实 Qwen tokenizer 逐 token 测试）；Python lint 通过。官方一致性检查：30 个 train、24 个 RL-train、6 个 internal-dev、20 个 test，50 个保留必须动作内容/格式与官方一致。
- 新增 GitHub Actions CPU 工作流，固定下载 Qwen tokenizer revision `cdbee75f17c01a7cc42f958dc650907174af0554`，不下载 4B 权重、不请求 DeepSeek、不使用 API key。远端是否通过应以该 commit 的 Actions 记录为准。
- 未执行：A800/vLLM 在线生成、DeepSeek user/judge API、真实 PPO 更新和逐 epoch 哈希、LoRA/base 参数变化、BF16 合并等价性、GPU 断点恢复与显存验收。

结论：代码可进入 GPU/API smoke，不能仅凭 CPU 测试启动正式 15 epoch RL 或声称训练有效。

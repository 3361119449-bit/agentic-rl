# Tau2 Airline：严格终局奖励与 ProCredit 逐轮分配方案

日期：2026-09-27

状态：用户已批准实施；实现与验证记录见配套实施计划

代码基线：378c062d84a90930d5b2422c440a39d6d6fec04a

本文冻结奖励定义、改动范围和验收标准；具体实现状态见 `../plans/2026-09-27-procredit-implementation.md`。

## 1. 目标与范围

将现有的多项加权训练奖励改为三个可解释的量：

1. R：是否满足完整的自定义成功条件。
2. Phi：当前满足了多少可验证的任务检查。
3. P：整条轨迹发生了多少已定义的过程错误。

通过终局分数比较同组轨迹，通过逐轮剩余任务回报分配轮次 credit。继续使用项目的 PPO 更新方式和有界动态采样，但重新定义“有效训练组”。

用户先要求制定方案，随后明确要求“请实施”。本文作为实施依据，不能视作已完成 GPU 或 API 验证。

保留以下实验边界：

- 模型、SFT 起点、agent 系统提示词、工具执行协议及 Tau2 官方判分独立于本次奖励改造。
- RL train 24 条、internal-dev 6 条、official test 20 条，以当前已提交划分为准。
- internal-dev 为 3、7、9、27、39、46；不重新选择开发集或训练任务。
- 官方 reward、自定义 strict_success、train_reward 分别保存；最终评估继续报告官方 pass^1/pass^4。
- 用户模拟器合规过滤、基础设施故障排除、独立环境、冻结评估 slot 和恢复身份继续有效。

## 2. 路线选择

| 备选路线 | 优点 | 代价与局限 | 决定 |
|---|---|---|---|
| 只改终局标量为 R + 0.5 Phi | 改动较小 | 所有轮次仍共享 advantage，未实现逐轮分配 | 仅作为后续消融，不作为主方案 |
| 终局分数 + 逐轮任务回报 + 轨迹级过程扣分 | 定义清楚，复用现有过程扣分，接近用户提供方案 | 需要贯通 rollout、队列、训练器和组过滤 | 推荐首版 |
| 同时将过程错误也分配到对应轮次 | 能直接定位错误轮次 | 还要定义扣分去重、封顶和跨轮分配，会扩大首版变量 | 首版不采用 |

首版只冻结一套默认算法，不做自动调参。保留旧奖励模式用于复现和回退，但不把旧配置字段混进新公式。

ProCredit 原文给出了终局进度、逐轮剩余回报和两层 advantage 的组合，实验设置为 c=0.5、gamma=1。本项目增加的 strict Judge、过程扣分、gate 和截断规则属于 Tau2 适配，不声称完整继承原论文的理论结论或实验收益。

来源：[ProCredit 方法与实验设置](https://arxiv.org/html/2609.27532v2#S3)。

## 3. 推荐冻结的首版默认值

| 项目 | 推荐值或规则 |
|---|---|
| 进度系数 c | 0.5 |
| 回报折扣 gamma | 1.0 |
| 逐轮项系数 lambda_turn | 1.0；只预留 lambda_turn=0 的消融 |
| 截断乘数 m | 当前定义的截断为 0.75，其余为 1.0 |
| 过程罚分 P | 复用现有去重规则、各项数值与 0.20 总上限 |
| 分数下界 | 扣分后裁剪到 0 |
| 分数上界 | 1+c=1.5；不裁剪到旧的 1.0 |
| Agent 违规 | 分数为 0，逐轮 advantage 为 0；保留轨迹级相对比较 |
| 逐轮中心化 | 同一采样组内，gate 通过且有策略 token 的所有轮次等权平均 |
| 轨迹标准差 | 总体标准差，epsilon=1e-6 |
| 组有效性阈值 | 有至少一个策略 token 的最终 advantage 绝对值大于 1e-8 |
| 组大小与更新规模 | 每组 8 条；每次更新 4 组，即 32 条 |
| 有界补采 | 每批 8 组，最多 3 批；原有计数和失败处理保留 |

这里的 epsilon、阈值、标准差定义均记录进恢复身份，避免实现时依赖库默认行为。

## 4. 完整数学定义

### 4.1 下标、分组和有效性

- i 表示轨迹。
- t=1,...,T_i 表示一次 assistant 生成；纯文本、工具调用、被本地拒绝的输出、未执行的截断输出都占用各自轮次。
- g 表示一次 prompt 采样产生的唯一 group uid，而不只是 task_id。同一个任务在不同候选批中出现时，不能把它们拼成一组。
- V_i 为已完成检查的 Agent 合规标志：必要的 policy/safety gate 和适用的转接合法性均通过时为 1，否则为 0。
- 用户模拟器违规、Judge/API pending、基础设施失败不定义为 V_i=0；它们沿用现有重采或未完成处理，不伪装成模型零分样本。

正文中的“合规轨迹”专指 V_i=1。禁用 gate 的配置开关与 gate 检查失败是两个概念。新模式要求必要 gate 开启，不允许通过关闭开关跳过本方案。

### 4.2 终局成功 R

普通任务沿用现有适用性语义：

R_i = AND(适用的 DB、COMMUNICATE、required_actions、judge_semantic 检查全部通过)。

转人工任务保留其专门条件：转接合法、必要前置动作完成、转接调用成功、必要沟通与转接语义检查通过。

约束：

- 不适用的分项不强制算失败；应存在的评分结果缺失则属于评分未完成。
- Judge 执行失败不能当作 semantic=false；未配置的 semantic 检查不能伪造为全部通过。
- 维持当前 strict_success 的含义及必要 gate；不因改奖励公式而放宽成功条件。
- R 与官方成功仍可能不同，包括外部截断时组件已满足但官方成功被审计为 0 的情况；继续分别保存并在分析中识别。
- 新模式训练要求 Agent reward Judge 开启。原有 no-reward-judge 路线仍属于独立的 official-only 基线，不暗中转成无 semantic 的 ProCredit。

### 4.3 固定检查集合与进度 Phi

普通任务候选检查为：

- 一个整体 DB 目标检查（仅在适用时）。
- 每条官方 COMMUNICATE 检查。
- 每条 required_action，保留参数匹配、一对一匹配和动作依赖约束。

转接任务使用与其任务目标对应的前置动作组、转接调用和沟通检查；semantic 仍只进入终局层。

每个任务先固定候选检查及稳定 check_id。在同一任务的固定初始状态上，选取适用且初始未满足的检查形成 C_g^+：

Phi_(i,t) = sum(C_k(s_(i,t)), k in C_g^+) / |C_g^+|。

若 C_g^+ 为空，则所有 Phi_(i,t)=0，退化为终局成功加过程扣分；不除以零，也不假设已经获得满额进度。

必须满足：

1. 分子只计算 C_g^+ 中的检查；分母在整条轨迹和同一组内固定。
2. 初始满足的检查只从进度中移除，仍保留适用的终局约束。
3. 同组的任务、初始状态和检查集合指纹必须一致；不一致时报错，不混合归一化。
4. DB 检查反映当前状态，允许 1 变回 0，不使用历史最大进度。
5. COMM 表示截至当前已交付的 assistant 文本中是否出现必要信息；REQ 表示截至当前历史中是否完成必要动作。二者本来就是历史性质的检查，不虚构为可撤销的 DB 状态。
6. 所有检查均按条目等权。冻结条目粒度，不随训练效果拆分/合并条目。
7. DB 与 REQ 可能描述同一目标的不同证据。首版保留二者并记录重叠情况，不把“等权”描述成“完全无重复奖励”。
8. 检查器、目标答案和进度值只用于训练评分，不追加到 agent prompt、工具结果或用户模拟器输入。

当前 train 标注文件覆盖官方 train 的 30 条任务，其中 allowed/required transfer 均为 false。因此首版当前训练集合使用普通任务检查集合；保留转接评分兼容和合成测试，不为制造训练信号而修改转接标注。

如果将来新增允许转接的任务，必须在运行前固定其检查定义与终局路由。不能因为模型中途选择了转人工，就换一套更小的进度分母。无法唯一确定目标的任务先拒绝新模式启动。

### 4.4 轨迹分数 S

S_i = V_i * max(0, m_i * (R_i + 0.5 * Phi_(i,T_i)) - P_i)。

决定如下：

- 截断仅折扣任务部分，不折扣过程罚分。
- 必要 gate 失败时 S_i=0，不给违规轨迹保留任务进度奖励。
- S 属于 [0,1.5]，不除以 1.5、不沿用旧的上界裁剪。
- 完整保留裁剪前分数以及实际发生的 P，便于检查零分区域被压平的比例。
- 合规低分样本可能因下界裁剪而变成同分，这是首版明确接受的取舍。

在首版固定的 c=0.5、m>=0.75、P<=0.20 下，合规 R=1 轨迹即使 Phi_T=0，其 S 也至少为 0.55；R=0 轨迹的 S 至多为 0.50。因此分数层仍优先完整成功。修改这些系数时必须重新检查此数值关系；这不是对加入组归一化后最优策略的理论保证。

例子：

| V | R | Phi_T | m | P | S |
|---|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 1 | 0 | 0 |
| 1 | 0 | 0.5 | 1 | 0 | 0.25 |
| 1 | 1 | 1 | 1 | 0 | 1.50 |
| 1 | 1 | 1 | 1 | 0.10 | 1.40 |
| 1 | 0 | 0.8 | 0.75 | 0.10 | 0.20 |
| 1 | 0 | 0 | 1 | 0.10 | 0 |
| 0 | 任意 | 任意 | 任意 | 任意 | 0 |

### 4.5 轨迹与逐轮 advantage

轨迹项在完整 8 条可用轨迹上计算：

A_i^traj = (S_i - mean_g(S)) / (std_population_g(S) + epsilon)。

当组内分数严格相等时，显式得到零轨迹项。

逐轮回报：

G_(i,t) = m_i * [R_i + 0.5 * (Phi_(i,T_i) - Phi_(i,t-1))]。

令 U_g^valid 为该组内 V_i=1 且该轮确有生成策略 token 的轮次集合：

bar_G_g = sum(G_(j,u), (j,u) in U_g^valid) / |U_g^valid|。

A_(i,t)^turn = V_i * (G_(i,t) - bar_G_g)。

若 U_g^valid 为空，所有逐轮项为 0。注意，屏蔽作用于中心化后的结果；不采用“先把违规 G 设为零，再和其他轨迹一起中心化”的做法。

最终：

A_(i,t) = A_i^traj + lambda_turn * A_(i,t)^turn。

数值与含义约束：

- bar_G 按轮次平均，不按 token 数量加权、不跨任务平均、不先按各轨迹平均后再平均。
- 逐轮项不除以标准差，也不在合并后额外进行一次 batch whitening。
- gamma=1，lambda_turn=1。
- 首版的 G 是剩余任务回报；P 只通过 A^traj 起作用。明确不称 G 为含所有扣分的完整 return-to-go。
- R 同时出现在两层是所选两层 credit 的定义，不另作去重。
- 对同一合规轨迹的相邻有效轮次，应满足 A_(i,t)-A_(i,t+1) = 0.5*m_i*(Phi_(i,t)-Phi_(i,t-1))。
- 查询本身的进度增量为 0，不代表其最终 advantage 必须为 0；前序查询仍可能因为后续成功得到 credit。
- G 和 advantage 允许为负；只给 S 设置非负下界。

违规轨迹的 A^turn=0；因为其 S=0 且所有组员的 S>=0，其 A^traj<=0。这解决了“违规零分反而超过合规负分”的排序问题。全组违规时所有 advantage 为零，组被判为无学习信号。

## 5. 逐轮检查如何得到

### 5.1 以对话前缀为依据

首版采用完整轨迹结束后对冻结前缀进行本地确定性检查的方式：

1. reset 后记录初始状态与初始消息边界。
2. 每次 assistant 生成及其环境交互结束后，记录真实交付消息的前缀边界和该轮实际工具事件。
3. 正常结束或外部截断时，在人工 cleanup 之前冻结完整交互及轮次边界。
4. 用户模拟器合规检查接受该轨迹后，对每个前缀计算相应检查值，再进行终局评分。
5. 用每轮检查值形成 Phi 序列、G 序列和可审计的 scoring inputs。

这与使用各中间状态检查的目标一致，但不新增逐轮 LLM Judge 请求。逐轮检查只能读取相应前缀，不允许用后续消息倒填前面的完成状态。

首版正确性路线：在独立评分环境中使用固定版本的官方 DB evaluator 对前缀重放，COMM 使用官方文本检查，REQ 使用本项目的前缀动作匹配。

DB 的官方比较涉及 agent 和 user 两侧数据库；只用当前已有的单侧 db_hash 或 db_changed 不足以替代它。任何初始化与参考动作执行只能发生在独立评分环境，不能触碰正在交互的环境。重复相同前缀可以缓存；不在首版先做未经对照验证的自制字段级 DB 评分。

来源：[固定 Tau2 版本的环境评分器](https://github.com/sierra-research/tau2-bench/blob/a2c024725189473d2d7cea3a5cfdbcc67478e41f/src/tau2/evaluator/evaluator_env.py)。

前缀重放可重用确定性结果，但不调用用户模型、agent 或 reward Judge。其实际耗时、内存及初始化一致性须在实施时测量。若成本过高，再用经逐前缀等价验证的状态快照/目标缓存优化；不默认降级成启发式进度。

### 5.2 特殊轮次

| 情况 | 进度与记录规则 |
|---|---|
| 一个 assistant 轮次内多个合法独立工具 | 所有实际结果作为同一轮的状态转移，只生成一个轮次 credit |
| 依赖工具被错误地放在同一轮 | 沿用动作依赖/政策判定；不以工具数量单独处罚 |
| 普通文本回复 | 可推进 COMM；REQ、DB按实际前缀检查 |
| 本地拒绝的非法输出 | 保留生成轮次和 token，未交付内容不计 COMM，未执行调用不计 REQ |
| 没有 EOS 的截断输出 | 不执行工具、不交付正文；该轮 Phi 沿用上一状态，仍保留策略 token |
| 超出预算/轮数后直接停止 | 不虚构新的 assistant 轮次 |
| 人工 cleanup | 不记成 agent 行为，不新增进度、轮次或成功证据 |
| 所有轮次均无策略 token | 保存审计，标为不可训练样本，完整组不参与更新；不制造 padding 轨迹凑满 |
| 评分异常或缺少前缀 | 保持评分失败/待处理；不把缺失检查补成 0 |

当前工具协议已允许合法多工具调用。新模式移除对“调用数量本身”的惩罚语义；旧 multiple_tool_calls 事件仅由旧模式读取和复现，不重新解释历史轨迹。

## 6. Token 对齐与 veRL 接入

### 6.1 Actor 侧记录

新增紧凑的逐轮字段，例如：

- progress_version、checkset_fingerprint、initial_state_fingerprint。
- progress_checks：check_id、类型、适用性、初始值、是否纳入 Phi。
- progress_trace：每轮 check bits、Phi_before/Phi_after、delta、真实消息前缀边界。
- response_turn_ids：与最终 response token 流等长，策略 token 标为对应 assistant_turn_index，其余标为 -1。
- terminal_success、gates、truncation_multiplier、process_penalty、raw_score、train_reward。
- 明确的 prompt group uid、policy_version、trajectory_id。

采样组 uid 必须来自真实采样器。如果 actor 接口未下发该值，由训练器另存原始组成员映射，不从 task_id 拼造 uid，也不覆写已原子落盘的轨迹。

数学轮次从 1 开始，现有 TokenTurn.assistant_turn_index 从 0 开始，ToolEvent.turn_id 使用其现有编号。通过显式映射转换；不把环境 turn_idx、工具事件 turn_id、生成轮次当成同一个编号。

策略输出的 EOS 继续保留在原始 token/log-prob 流内并分配该轮 advantage。工具结果、用户文本、模板插入 token 和 padding 的 mask=0、advantage=0。只更换其训练优势值，不重编码策略生成 token。

### 6.2 训练器侧计算

在项目已有的 CappedPPOTrainerSync 扩展中加入独立的新模式路径：

1. 从 TransferQueue 读取标量分数、逐轮任务回报、gate 和 token-to-turn 映射。
2. 按采样 uid 划分完整组，校验版本、task 和检查集合一致性。
3. 使用与组过滤相同的纯函数计算两层 advantage。
4. 将每轮最终 advantage 广播到本轮策略 token。
5. 写入 veRL 实际消费的 advantages；首版无 Critic，兼容字段 returns 写入相同的最终 advantage。真正的逐轮 G 单独保存，不从 returns 指标读取。
6. 两个 PPO epoch 使用同一份 rollout old log-prob 与同一份预先计算的 advantage。

训练器为每个完整候选组另存原子化组审计记录：8 个成员及槽位、版本/检查指纹、S 的均值与标准差、有效轮次均值、两层 advantage、保留/丢弃决定。丢弃组也记录；离线重算不得把别的采样组补进来。检查下游消费者未把 returns 当作 Critic 目标，是实际依赖契约验收的一部分。

固定 veRL 版本在训练器的 _compute_advantage 阶段构造并写回 advantages。本设计选择在该项目扩展点接入，避免修改外部 veRL checkout。具体队列字段的序列化和 nested/padded 布局必须由真实依赖契约测试确认。

来源：[固定 veRL 版本的 trainer_base](https://github.com/verl-project/verl/blob/483b8a009ba3a97563edee3a19887e4862b8094a/verl/trainer/ppo/v1/trainer_base.py)。

不通过把 progress delta 直接写成 token reward 然后继续调用旧 GRPO 来冒充逐轮 ProCredit；旧路径可能仍将整条奖励求和后广播。

## 7. 有界动态采样的新定义

新模式中的“有信号组”：

存在某个 response_mask=1 的 token，使最终 |advantage| > 1e-8。

检测使用完整组的最终 advantage，与优化器实际收到的数值一致；不只检查 S 的方差，也不只检查 Phi_T。

典型情况：

- 同组全部成功且 S 相同，但 G 随轮次变化：保留。
- 同组全部失败且最终进度相同，但仍有逐轮进度变化：保留。
- 仅过程扣分不同，导致 S 不同：保留。
- S 相同且所有有效 G 相同：丢弃。
- 全组违规且没有任何 advantage：丢弃。
- 7 条完成、1 条基础设施失败：仍整组丢弃，不能只用 7 条。

实现时必须绕开父类提前按“标量 reward 恒定”驱逐组的逻辑，否则后续新检测已无样本可用。新路径不向旧过滤器伪造一个有方差的奖励字段。

保留每批 8 组、每组 8 条、最多 3 批，收满 4 个有信号组才更新的规则。上限为每次候选尝试 192 个合规候选槽位；用户模拟器重采的物理轨迹另计，不能把 192 描述成整个实验或所有重试的硬上限。

不足 4 组时只增加 attempt_step，不更新参数、学习率、optimizer_step；现有连续跳过上限继续有效。

新增审计区分：constant_score_with_turn_signal、all_zero_advantage、infrastructure_failed、invalid_group_identity 等。成功组统计使用 R/官方成功标志，不再用 score==1 代表成功。

## 8. 拟修改的模块

下列路径均相对于仓库根目录；新增名字为建议，可在正式实现计划中细化。

| 模块 | 职责 |
|---|---|
| agentic_rl/src/tau2_agentic_rl/reward/progress.py（新增） | 固定检查集合、前缀评估、Phi 及指纹 |
| agentic_rl/src/tau2_agentic_rl/advantages.py（新增） | 两层 advantage、中心化、组信号检测的共享纯函数 |
| agentic_rl/src/tau2_agentic_rl/reward/score.py | 分离旧/新公式，保存裁剪前后结果 |
| agentic_rl/src/tau2_agentic_rl/reward/required_actions.py | 复用前缀匹配和依赖检查，不放宽动作完成条件 |
| agentic_rl/src/tau2_agentic_rl/reward/transfer_branch.py | 保留转接终局语义，提供适用检查的适配 |
| agentic_rl/src/tau2_agentic_rl/schemas.py | 新轨迹版本、逐轮审计结构及新分数范围 |
| agentic_rl/src/tau2_agentic_rl/environment/tau2_gym.py | 提供真实前缀边界/必要冻结输入，保留环境隔离 |
| agentic_rl/src/tau2_agentic_rl/agent_loop/airline.py | 记录轮次、传递新奖励字段、对齐 token |
| agentic_rl/src/tau2_agentic_rl/verl_capped_trainer.py | 新组过滤与真实 advantage 写回 |
| agentic_rl/src/tau2_agentic_rl/training_config.py、config.py | 新模式映射、互斥校验和启动前检查 |
| agentic_rl/src/tau2_agentic_rl/rl_resume.py、storage.py | 算法/数据/检查器身份及新旧记录读取 |
| agentic_rl/scripts/rescore_saved_trajectories.py | 对新完整记录重算分数与组级 credit；缺输入则拒绝 |
| agentic_rl/configs/rl/airline_procredit_v1.yaml（新增） | 独立的新实验配置，保留旧配置 |
| agentic_rl/tests/ 与 contract_tests/ | 数学、环境边界、队列、PPO 和恢复验证 |
| agentic_rl/README.md、REVIEW_FOLLOWUP.md | 运行方法、指标定义、兼容性和验证范围 |

## 9. 拟议配置与身份隔离

以下是配置意图示例，尚不是可以直接运行的配置：

    reward:
      mode: strict_progress_v1
      reward_version: v5-procredit-draft
      progress_scale: 0.5
      score_floor: 0.0
      truncation_multiplier: 0.75
      penalty_placement: outside_truncation
      mandatory_policy_gate: true
      task_safety_gate: true
    credit:
      mode: procredit_turn
      version: procredit-turn-v1
      gamma: 1.0
      turn_coefficient: 1.0
      turn_centering: valid_turn_mean
      trajectory_std: population
      epsilon: 0.000001
      process_penalty_in_turn_return: false
    dynamic_sampling:
      criterion: final_advantage_nonzero
      signal_tolerance: 0.00000001

版本号在实施前检查是否已被占用；正式版本不能含 draft。示例不覆盖现有配置文件。

新模式拒绝同时出现旧 normal_weights、transfer_weights、progress_coefficient、strict_success_coefficient，避免配置看似生效但被悄悄忽略。旧模式继续使用原有字段。

恢复身份包含奖励公式、所有系数、进度检查器版本、检查集合与标注指纹、轮次定义、标准差/中心化方式、过滤准则和代码身份。

新实验默认从同一个已验证 SFT 合并模型启动。旧 RL checkpoint 可作为另行标记的权重初始化实验，但不得携带旧优化器状态冒充同一实验续训。

新记录需要显式版本；旧记录保持可读、可按旧公式复现。缺少完整前缀、对应初始化输入、组成员或 token 映射的旧轨迹，不能仅从最终分数“补造”逐轮 credit。旧样本也不能自动注入新的 on-policy 更新。

## 10. 实施顺序与阶段交付

本节只是后续实施路线；本次不执行。

| 阶段 | 工作 | 交付与进入下一阶段的条件 |
|---|---|---|
| 0. 固定定义和依赖边界 | 在固定 Tau2/veRL 上验证前缀评分、初始化、队列字段和训练器扩展点 | 确认无 live DB 修改、无额外 LLM 请求；有真实接口契约证据 |
| 1. 评分与 credit 内核 | 实现检查集合、S、G、两层 advantage、gate 与组信号函数 | 表格与手算样例一致；边界案例通过 |
| 2. Rollout 记录 | 增加真实前缀边界、token-to-turn 映射和新记录结构 | 所有策略 token 恰好映射一次，cleanup/未交付文本不产生进度 |
| 3. 队列与训练器 | 接通字段、替换组过滤、写入实际 token advantage | 等分但有逐轮信号的组可到达 actor；无信号组不更新 |
| 4. 配置与复现 | 独立配置、模式校验、恢复指纹、离线审计、旧模式兼容 | 新旧实验不能误混；旧评估口径不变 |
| 5. 离线及真实依赖验证 | 运行目标测试和必要回归，检查真实 pinned veRL 数据布局 | 区分纯 Python、真实依赖和 GPU 验证的完成状态 |
| 6. 非 test GPU smoke | 用现有 train smoke 集验证一个实际更新及恢复 | 确认优势值进入 loss，PPO 不变式成立，记录开销 |
| 7. 冻结正式实验 | 审查 train/internal-dev 的进度质量和统计，再冻结配置 | 确定实验预算与停止规则后才进行正式训练及 test |

阶段 0 若发现前缀重放或队列能力与设计不符，应先修订设计；不得用虚构进度、关闭必要校验或退回标量广播来凑通训练。

## 11. 验收标准

### 11.1 数学与边界

- 第 4.4 节所有数值例子精确符合公式。
- 同组分数恒定时轨迹项为 0；总体标准差定义与配置一致。
- 长短轨迹的中心化按真实轮次计数，不按 token 数量或 padding 计数。
- 无合规轨迹、单条合规轨迹、空检查集合均产生有限数值。
- V=0 的轨迹逐轮项为 0，最终 advantage 不为正。
- DB 进度回退产生负 delta，且相邻轮次 advantage 差值符合定义。
- 仅改变 P 会改变 S/轨迹项，不改变 G/逐轮项。
- lambda_turn=0 精确退化为新标量奖励基线；不冒称回到旧加权奖励。

### 11.2 进度检查

- 初始已满足检查不进入分子分母，但终局仍检查。
- 查询不因调用成功或返回大量文本就获得 DB 进度。
- 必要动作要求真实成功，写操作仍要求实际效果；重复调用不重复增加完成数。
- 同一轮中有依赖的动作不因实际执行顺序碰巧正确就算跨轮依赖满足。
- 合法多工具调用只有一个 assistant 轮次，不按工具数量额外奖励或处罚。
- COMM 只检查真实交付的 assistant 文本；用户重复答案、工具返回答案、本地拒绝文本都不能替代它。
- 完整前缀的 DB/COMM 检查与固定官方评分器的对应组件一致。
- 终局进度不包含 cleanup；评分器不改变源环境、源轨迹或用户模型缓存。
- 转接适配用合成样例验证，不更改现有训练任务标注。

### 11.3 训练链路

- 真实 veRL/TransferQueue 往返后，新增字段保持内容和组身份。
- 最终 advantages 与 response_mask、token-to-turn 映射逐 token 对齐。
- 用户/工具/模板/padding advantage 为 0；生成 EOS 与本轮其他策略 token 使用同一值。
- 全成功同分组、全失败同分但有逐轮变化的组均能通过新过滤。
- 相同 task_id 的不同采样 uid 不相互合并。
- 含基础设施失败的组不补假样本，用户违规重采不混入旧尝试。
- 达补采上限且不足 4 个有信号组时无 optimizer step，计数与 checkpoint 行为正确。
- 两个 PPO epoch 的 old log-prob 和 advantage 均冻结，更新前 ratio 在数值容差内接近 1。
- 真实 actor 接收非均匀逐轮 advantage；测试不能只验证日志或辅助纯函数。

### 11.4 兼容与评估

- 新分数 1.5 不被 Pydantic 拒绝、不被下游截断成 1.0。
- 官方 reward 与 pass^k 不使用新 S，也不把 S==1 当成成功。
- 旧奖励配置仍可按原逻辑运行；新模式错误混入旧字段时启动失败。
- 新旧 reward/credit/checkset 身份不一致时无法恢复优化器。
- 新记录离线重算与在线结果一致，不重复乘截断系数；组级 advantage 必须有完整原始组。
- 官方 test 的 80 条有效样本要求与用户模拟器过滤口径继续适用；不依据 test 结果修改设计。

## 12. 监控、实验与已知取舍

至少记录：

- R、官方 reward、Phi_T、P、S、裁剪前分数及下界裁剪比例。
- 每类检查数量、初始剔除数量、每轮 delta、DB 回退与任务间进度分布。
- gate 失败率、Judge pending、用户违规拒绝和重采次数。
- A^traj/A^turn 的分布、最终非零比例、两部分相对尺度。
- 同分但有逐轮信号的组数、无信号组数、补采批数、attempt/optimizer step。
- 前缀评分耗时、队列载荷与内存、rollout tokens、GPU 更新耗时。

重要取舍：

1. 下界裁剪会压平零分区域的过程错误差异；用裁剪比例和错误率评估，不在 test 上调整。
2. 条目等权让检查数量决定类别占比；DB 与 REQ 的重叠作为设计事实披露。
3. 当前转接标注在 train 中没有正例；本方案不能凭测试覆盖声称学到了转接能力。
4. G 不含过程罚分，逐轮项可能部分抵消整条轨迹的负 advantage；需要监控，但这属于所选两层目标的行为，不自动定义为实现错误。
5. 非均匀进度不保证与真实任务难度完全一致，必须人工抽查 train/internal-dev 的代表性前缀。
6. 只在组内 advantage 出现非零值时保留样本，仍是一种训练分布选择；记录保留率，不能声称与无过滤 ProCredit 完全相同。
7. 终局裁剪、gate、额外 semantic 和截断折扣使本方案不同于论文原始设置，不作最优策略不变或必然提升的承诺。

建议验证顺序：先离线检查完整的 24 条 train 与 6 条 internal-dev 检查定义，再对非 test 代表性轨迹进行前缀人工核查，然后完成小规模 GPU smoke。

若资源允许，正式比较至少包含：

- 当前旧奖励与旧训练方法。
- 新 S、lambda_turn=0 的标量基线。
- 新 S、lambda_turn=1 的完整方案。

每组使用相同 SFT 起点、任务划分、采样参数、用户模拟器与 Judge 设置，并同时报告 optimizer steps、生成轨迹/token 和计算开销。一个实验版本只能验证工程可用与观察效果，不能单独证明收益来自逐轮 credit。

正式 GPU 预算、训练步数、重复 seed 数和停止规则尚未由用户指定，不在本方案中虚构；它们需要在实际运行前冻结。这里不影响先审阅算法与工程设计。

## 13. 本轮完成边界

本轮只新增本设计文档。未修改 reward、actor、trainer、配置、任务标注或数据集；未安装依赖、运行训练、请求付费模型或提交 Git。

后续进入实现前，应先确认这份设计中的首版默认选择，再细化可执行的实现计划。尤其需要审阅：非负分数下界、过程罚分只留在轨迹层、合规轮次中心化、根据最终 advantage 过滤组、独立的新实验配置。

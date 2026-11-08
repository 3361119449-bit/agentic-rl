#35轮上限、截断归零与有界重采
当前默认奖励标识为 v12-procredit-35turn-zero-truncation；截断策略为 zero-trajectory-v1。
此前v11配置保存在 configs/rl/airline_procredit_v4_v11.yaml；
旧标量GRPO配置保存在 configs/rl/airline_grpo_v1_v4.yaml。

训练、内部验证和正式测评均最多生成35个助手轮次。
工具调用与回复各算一次助手生成；用户消息、工具回执不计。
第35轮自然完成仍按正常结束计分；35轮后未完成则不再生成第36轮，
记录hard_turn_limit。35轮内没有旧的第16轮起超轮成本。
上下文/生成预算耗尽、生成截断、原生max_steps、context_window_exceeded
与硬轮次截断采用相同归零规则；已有转人工工具调用不能覆盖真正的截断原因。

##训练重采
只在train槽位启用max_truncation_resamples=2。
第一次截断后只替换该槽位，最多再采2次；同一策略权重租约、组和任务保持不变。
每次重采有新trajectory_id，保存parent、全部截断attempt ID、原因和已用重采次数。
2次截断重采都耗尽时，最后一条仍保留为截断且所有训练信号为0。
接口故障、用户模拟无效与截断的替换预算分别计数且均有界，不互相重置。
因此只有截断时是初采+最多2次重采；混合其他故障时，物理调用数还包含独立的故障预算。
rollout_attempts.truncation计实际启动的截断重采；
slot_recovery.truncated_attempts记录本槽位所有已观察到的截断。

##归零范围与审计
截断轨迹的有效总奖励、严格成功、进度信用、逐轮与token优势全部为0。
最终归零优先于错误负优势、正确拒绝奖励、组内中心化和其他局部信用；
非截断轨迹继续采用原ProCredit-v4和Target-v2规则。
原始token、log-prob、mask、工具回执和实际进度保留审计；
读取训练信用时重新核对冻结输入哈希、终止原因与token映射。
截断可确定为0，不再调用语义/政策Judge或用户合规Judge，不伪造通过标签。

##测评
内部验证及独立测评均不因截断重采。截断保留在有效样本分母，按失败计0。
新口径主指标是evaluation_pass1/evaluation_pass4；
raw_tau2_official_pass1/raw_tau2_official_pass4单列原始Tau2分供核对。
四条样本的pass4依然要求四条全成功。
原始官方分和有效测评分分别保存；旧版本可能已覆盖截断官方分，不能据旧记录补造原分。
新manifest绑定截断策略和代码/config哈希；使用新run/tag，不冒充旧实验续跑。

本次没有启动rollout或GPU训练，历史80条记录、奖励包与SFT检查点不改写。

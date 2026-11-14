# Tau2 Airline 五任务修复版

适用的 Tau2 提交是 `a2c024725189473d2d7cea3a5cfdbcc67478e41f`。`tasks.json` 包含五个完整的 Tau2 任务定义。使用 `score.py` 对同一条原生 `SimulationRun` 分别计算官方原任务分和修复分。文件保留原任务 ID，便于对照。

| 任务 | 修复 |
| --- | --- |
| 7 | 第三次客服消息之后询问其他未取消的未来订单。以通话开始时原已付款项为准，排除正在处理的 `XEHM4B` 和 `59XX6W`，也排除过去行程。初始数据库中符合条件的是 `7WPL39` 的 $402 和 `3EMQJ6` 的 $306，共 $708。 |
| 14 | 对本次新预订的付款流水按付款 ID 和金额排序后比较数据库。卡号、金额、航段顺序、取消和余额仍参与比较。 |
| 21 | 用户指定用一张足额礼品卡支付已确认的改签费用，并在足额卡中选择余额最小的。差价为 $59，最小的 $6 卡不足，足额卡中应选 $113 的卡。官方参考动作的卡号已符合这个要求。 |
| 23 | 对新订票的付款流水和系统生成的预订号消除执行顺序的影响。按完整订单内容匹配，保留乘客、航段、付款来源和金额之间的对应关系，并同步比较用户持有的预订。已有预订号保留。 |
| 39 | 将用户取消原因固定为计划变更。保留 `8C8K4E` 和 `LU15PA` 的取消要求，移除 `MSJ4OA` 的取消动作。后者的保险不覆盖计划变更，订单超过 24 小时，两段航班均未被航司取消，舱位为经济舱。 |

Task 14 和 23 的任务 JSON 与原版相同，DB 修复由 `score.py` 执行。Task 7 沿用官方 `COMMUNICATE` 字符串检查，因此仍需用语义评审识别数字出现在错误语境里的情况。Task 21 的修改适用于之后按新场景生成的对话，历史授权是否充分仍按原对话判断。

## 验证结果

7 项 pytest 回归通过。对照使用相同的工具和真实回执，评分时在独立 Tau2 环境严格重放。测试包含 Task 7 报错总额和漏取消，Task 14 漏取消和扣款分配错误，Task 21 选错卡，Task 23 漏订和证书归属错误，以及 Task 39 多取消和漏取消。截断案例仍为零分。

使用 SFT v4 包的 `audit/all_reviewed_trajectories_do_not_train.jsonl` 对这五个任务共 23 条来源做严格重放。历史记录沿用 SFT 审计的 `user_stop` 假设，并使用与历史回执匹配的 Pydantic 2.14.2。以下分数仅表示 DB 和 COMMUNICATE 检查结果。

| 任务 | 历史来源 | 原分 → 修复分 |
| --- | ---: | --- |
| 7 | 5 | 5 条 `0 → 1`。 |
| 14 | 5 | 5 条 `1 → 1`。对照重排礼品卡时为 `0 → 1`。 |
| 21 | 5 | 5 条 `1 → 1`。其中两条历史来源因授权不足已被 SFT 隔离，终态分没有体现这点。 |
| 23 | 4 | 4 条 `1 → 1`。对照调换 Raj 与 Liam 的订票顺序时为 `0 → 1`。 |
| 39 | 4 | 仅取消两张订单的 `airline_39_trial_4_seed_647860` 为 `0 → 1`，仍有未闭环的主管请求。另三条在不同原因的历史对话里取消了 `MSJ4OA`，在新的计划变更场景下为 `1 → 0`。 |

## 复现

在 `agentic_rl` 目录下，用安装了固定版本 Tau2 和 pytest 的 Python 运行。以下 PowerShell 示例使用本机的 Tau2 数据目录。

```powershell
$env:PYTHONPATH = 'D:/agentic-rl-deps/tau2-bench-data/src'
$env:TAU2_DATA_DIR = 'D:/agentic-rl-deps/tau2-bench-data/data'
$env:LITELLM_LOCAL_MODEL_COST_MAP = 'True'
python -m pytest task_repairs/test_repairs.py
python -m task_repairs.score path/to/simulation_run.json
```

这两条命令在本地运行工具和评分，不需要模型 API。

正式 RL 使用 `configs/rl/airline_procredit_v4_train30.yaml`。默认启动入口和 `--stage full_train` 都选用该配置，按固定 Tau2 官方训练列表训练全部 30 个任务。先运行 `python scripts/prepare_tau2_dataset.py` 准备 Parquet。修复定义用于训练用户模拟、整轨 Judge、终局 DB/COMM 和每轮进度。Task 7 使用原付款小计 $708。Task 39 的必要动作和语义检查均已更新。Task 14、23 的终局和进度共用修复版 DB 比较。

`judge.enabled=true` 为每条符合评分条件的训练轨迹启用整轨 Judge，并加载各任务的指导内容。ProCredit 仍按现有配置取得逐轮标签。正式 30 任务训练关闭内部验证回合，旧的 6 个内部验证任务已编入训练。正式训练可用 `--dry-run` 检查训练任务列表和启动参数。

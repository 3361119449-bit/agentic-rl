# 用已保存的 smoke 结果更新 actor

这是独立入口，只新增文件，不修改 `train_airline_grpo.py`、`verl_entrypoint.py`、rollout、reward 或 `ppo_audit.py`。辅助模块全部放在 `scripts/saved_rollout_support/`，不向 `src/` 添加文件，因此旧训练的源码 resume 指纹也保持不变。普通训练仍按原入口运行。

它读取结果包中的原始 token ID、vLLM rollout log-prob、group audit 和逐 token advantage，只创建 `role="actor"` 的 veRL worker，执行一次外层 PPO update，随后保存 FSDP/LoRA checkpoint。不会启动 vLLM server、用户模拟器、Judge、训练数据采样或验证 rollout，也不需要 DeepSeek API key。

## 先检查数据

在 `agentic_rl` 目录、原训练 Python 环境中运行；项目包应已安装（或设置 `PYTHONPATH=src`）：

```bash
python scripts/update_saved_rollouts.py \
  --results /path/to/m2_procredit_v4_300513e_a800a_smoke_seed42_results.tar.gz \
  --dry-run
```

`--results` 也可以是解压后的运行目录。直接读取压缩包时不会解压到磁盘。`--manifest /path/to/report.json` 可以保存检查结果。

这份结果包的预期检查结果是：8 个有效组、62 条轨迹、85,834 个策略 token、2 条零 mask 填充行、PPO mini-batch 为 32、PPO epochs 为 2。这相当于一次外层 update 内的 4 次 optimizer iteration；不重新做 advantage normalization。

dry-run 没有提供 `--model-path` 时只检查数据，报告中 `base_model_verified=false`。加上原模型目录会同时核对权重、配置、tokenizer 与 chat template 的 SHA256。

## 执行更新

仍使用原 smoke 的 CUDA/veRL Python 环境，以及固定 veRL commit `483b8a009ba3a97563edee3a19887e4862b8094a`。模型必须是原始合并后的 SFT 基础模型，不能换成已经做过 RL 更新的 adapter/model。

```bash
python scripts/update_saved_rollouts.py \
  --results /path/to/m2_procredit_v4_300513e_a800a_smoke_seed42_results.tar.gz \
  --model-path /root/autodl-tmp/sft_eval/models/areal_sft_epoch2_merged \
  --verl-root /path/to/your/pinned/verl \
  --output-dir /root/autodl-tmp/saved_smoke_update_seed42
```

输出目录必须不存在，防止覆盖旧实验或并发写入。代码只支持这类 `policy_version=0`、ProCredit v4、单 GPU、初始 LoRA 的已审计批次；不能把任意历史 rollout 当作当前模型的在线训练数据。

结果写入：

- `checkpoints/global_step_1/actor/`：原生 veRL checkpoint，可使用现有 `scripts/export_verl_lora.py --stage rl ...` 导出 PEFT adapter。
- `replay_manifest.json`、`actor_config.yaml`：数据来源、轨迹成员、恢复方式、批次指纹和实际训练配置。
- `ppo_audit.json`：沿用分支上修复后的审计，actor/rollout 数值差异是诊断信息，不以 `0.005` 阈值阻断更新。
- `worker_audit.json`：worker 内部检查 old log-prob、advantage、mask 不变，可训练参数确实变化，以及所有 optimizer iteration 的 loss/grad_norm 有限且未缺失。
- `metrics.json`、`update_status.json`：训练指标和状态。只有 checkpoint 保存完成后，状态才是 `complete`。

这是从保存的批次做一次独立更新，不恢复原 trainer 的 sampler/dataloader 状态。继续正常训练时，应先导出/合并这个 checkpoint，再按现有流程以权重开始新的运行；这个目录不冒充完整的原训练 resume checkpoint。

## 恢复数据的边界

每轮生成都保存了原始 prompt/output token ID。脚本逐轮核对生成时的 prefix、输出、turn mapping、log-prob 数量和 frozen reward evidence。旧概率一直作为 PPO 分母；actor 重算仅用于审计，不替换旧概率，也不重新调用任何评分 API。

原结果包没有保存最后一次生成后部分观察文本的 token ID。脚本只裁掉这段尾部，并要求其全部 `mask=0`、advantage 为零；这份包总计裁掉 2,419 个 token，保留全部 85,834 个策略 token及其因果上下文。组内原 session 顺序保持；原 TransferQueue 的组间顺序未保存，因此按 group uid 排序。这是复用同一批经验，不能保证与原队列的逐 bit 训练结果一致。

## 验证

普通 CPU 测试：

```bash
python -m pytest -p no:cacheprovider tests/test_saved_rollout_update.py
```

有 PyTorch 和 TensorDict 的环境可运行 `tests/test_saved_rollout_tensors.py`。设置 `VERL_REPLAY_TEST_ROOT` 为固定 veRL checkout、`SAVED_ROLLOUT_TEST_RESULTS` 为结果包路径后，可运行 `tests/test_saved_rollout_verl_contract.py`，检查真实配置和原生 response slicing。

开发验证包含真实结果包 dry-run 和 CPU 数据/张量检查。本地没有 A800/CUDA 训练环境，实际模型更新和 checkpoint 导出仍需在原 GPU 环境运行验证。

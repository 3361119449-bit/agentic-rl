# AReaL airline SFT 严格泄露清洗流程

## 输入与目标

- 输入：`airline_sft_no_thinking_under_16k.jsonl`
- 清洗清单：综合泄露审计中的 `recommended_strict_primary_cleanup`
- 删除范围：90 个 `source_dialog_id` 的全部前缀样本，共 1,198 行
- 预期输出：10,649 行、909 个源对话

## 本次实际结果

- 输入：11,847 行、999 个源对话
- 删除：1,198 行、90 个源对话
- 输出：10,649 行、909 个源对话
- 删除清单残留：0
- 输出 JSONL 有效性：通过
- 保留行：原始字节复制，没有重新序列化
- 输入 SHA-256：`0248acf2d4aa7e8b90603c92940cd5fbd1fb050a9c76518cc6836cae58685aa2`
- 输出 SHA-256：`5535171a7b6f271d282966ffcb5d5b1a7a5581bc2aa7226667624d9e20ed4cb0`

清洗完成后重新运行综合泄露审计，以下目标全部归零：

- 完全相同的 Tau2 test 参考查询：0
- 相同测试目标航班/日期的工具调用或输出：0
- Test 13 目的地修改拒绝模板：0
- Test 32 Basic Economy 两步修改模板：0
- Test 48 “声称刚订票但已超过 24 小时”模板：0

复扫仍保留 2 个“支付 ID 仅被读取展示”的低风险事件、22 个测试初始航班状态事件，以及 28 个与初始航线有关的日期搜索事件。这些不属于本次严格主清洗定义中的测试目标/答案泄露；如需研究最严格状态去记忆，可另做 `initial_state_ablation_only` 或 `cross_date_static_schedule_ablation_only` 消融。

这里的“严格”集合是以下两部分的并集：

1. 动态环境污染：测试参考查询完全相同，或测试目标航班在同一天出现在 SFT 工具调用/输出中。
2. 强任务模板污染：人工复核确认的 Test 13、32、48 高相似决策分支。

## 清洗原则

1. 只根据 `metadata.source_dialog_id` 过滤，不修改任意消息、工具调用、答案或 metadata 字段。
2. 一个源对话会生成多个长短不同的前缀样本；只要源对话被命中，就删除它的所有前缀行。
3. 保留行按原顺序、原始字节直接复制，不重新序列化 JSON，避免无关格式变化。
4. 输入文件与输出文件必须不同，输出放在独立文件夹，不覆盖原数据。

## 实际处理步骤

1. 读取审计清单 `tau2_areal_leakage_removal_manifest.json`。
2. 选择 `recommended_strict_primary_cleanup.dialog_ids`。
3. 流式读取输入 JSONL，并解析每行的 `metadata.source_dialog_id`。
4. 命中清单的行全部跳过；其他行按原始字节写入临时文件。
5. 校验清单中的 90 个 ID 在输入中全部存在，且删除行数严格等于 1,198。
6. 重新读取临时输出，检查每行 JSON 有效、目标 ID 残留为 0、输出行数一致。
7. 校验成功后原子地移动为最终 JSONL，并保存 SHA-256、行数、源对话数和逐对话删除行数。

## 复现命令

从 `strict_leakage_cleaned` 根目录运行：

```powershell
python ".\code\filter_sft_by_leakage_manifest.py" `
  --input "..\airline_sft_no_thinking_under_16k.jsonl" `
  --manifest ".\manifest\tau2_areal_leakage_removal_manifest.json" `
  --set-name "recommended_strict_primary_cleanup" `
  --output ".\data\airline_sft_no_thinking_under_16k_strict_leakage_cleaned.jsonl" `
  --report ".\verification\cleaning_verification.json"
```

## 输出文件

```text
strict_leakage_cleaned/
├── data/
│   └── airline_sft_no_thinking_under_16k_strict_leakage_cleaned.jsonl
├── code/
│   └── filter_sft_by_leakage_manifest.py
├── manifest/
│   └── tau2_areal_leakage_removal_manifest.json
├── docs/
│   └── CLEANING_PROCESS.md
└── verification/
    └── cleaning_verification.json
```

- `data`：清洗后的训练数据。
- `code`：可复现清洗脚本。
- `manifest`：完整删除 ID 和不同清洗层级。
- `verification`：输入、输出 SHA-256、行数、源对话数及残留检查。
- `docs`：本流程说明。

此步骤不会重新执行 thinking/reasoning 清理或 16k tokenizer 过滤；它直接以已经完成这两步的 11,847 行文件为输入，只追加泄露过滤。

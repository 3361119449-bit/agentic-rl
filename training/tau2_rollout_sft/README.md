# Tau2 train rollout SFT and pass^1/pass^4 evaluation

This directory implements the second SFT data source and the four requested
ablations:

| experiment | first SFT stage | second SFT stage |
|---|---|---|
| AReaL only | cleaned AReaL data | none |
| AReaL then Tau2 | cleaned AReaL data | successful Tau2-train rollouts |
| AReaL then matched AReaL | cleaned AReaL data | random AReaL rows, count-matched to Tau2 |
| Tau2 only | successful Tau2-train rollouts | none |

Final evaluation reports only official Tau2 `pass^1` and `pass^4` on the
airline **test** split. In Tau2, `pass^4` means that all four sampled trials for
a task succeed; it is not the usual “at least one of four succeeds” pass@4.

## Pinned components

- Tau2 package version: `1.0.1`, pinned inspected commit
  `a2c024725189473d2d7cea3a5cfdbcc67478e41f`
- Rollout agent: `deepseek/deepseek-v4-flash`
- User simulator: `deepseek/deepseek-v4-pro`
- Student: `Qwen/Qwen3-4B-Instruct-2507`
- Context limit: 16,384 Qwen tokens

DeepSeek thinking is explicitly disabled for both rollout participants. The
converter also rejects a simulation if any non-empty `thinking`, `reasoning`,
`reasoning_content`, or `<think>` material is found.

## 1. Install official Tau2

Keep Tau2 separate from this repository:

```bash
git clone https://github.com/sierra-research/tau2-bench.git /root/tau2-bench
cd /root/tau2-bench
git checkout a2c024725189473d2d7cea3a5cfdbcc67478e41f
uv sync
```

Set the API key in the environment. Do not put it in a command, JSON file, or
Git commit:

```bash
export DEEPSEEK_API_KEY='your-key'
```

## 2. Export the exact agent context

The trajectory file does not contain the agent system message or tool schemas.
Export them from the same pinned Tau2 checkout that will generate the rollouts:

```bash
cd /root/tau2-bench
uv run python /root/agentic-rl/training/tau2_rollout_sft/export_tau2_airline_context.py \
  --output data/simulations/airline_agent_context_v1.0.1.json
```

This exports only the public airline policy/system prompt and agent tool
schemas. It does not export task instructions, evaluation criteria, or database
state.

## 3. Roll out official train tasks

The default is four trials for each of the 30 official train tasks. First use a
one-task smoke test, then start or resume the full run:

```bash
cd /root/agentic-rl

python training/tau2_rollout_sft/run_tau2_deepseek.py \
  --tau2-dir /root/tau2-bench \
  --split train \
  --save-name airline_train_deepseek_flash_pro_smoke \
  --num-tasks 1

python training/tau2_rollout_sft/run_tau2_deepseek.py \
  --tau2-dir /root/tau2-bench \
  --split train \
  --save-name airline_train_deepseek_flash_pro_4trials \
  --auto-resume
```

The launcher uses Tau2's `llm_agent`, `user_simulator`, official `train` split,
communication-protocol enforcement, and API keys from the environment. Results
are written by Tau2 under:

```text
/root/tau2-bench/data/simulations/<save-name>/results.json
```

Use `--dry-run` to inspect a command without spending API credits.
The launcher refuses an unexpected Tau2 commit unless
`--allow-unpinned-tau2` is explicitly supplied. The converter also refuses a
results/context commit mismatch.

## 4. Convert successful rollouts to SFT

Install the Qwen tokenizer in the environment running the converter:

```bash
python -m pip install "transformers==4.57.1"
```

This matches the SFT environment. Conversion is also regression-tested with
Transformers 5.10.4: it explicitly requests token IDs (`return_dict=False`),
so the length is not the number of fields in a `BatchEncoding`.

Convert the full run:

```bash
python training/tau2_rollout_sft/convert_tau2_results_to_sft.py \
  --results /root/tau2-bench/data/simulations/airline_train_deepseek_flash_pro_4trials/results.json \
  --context /root/tau2-bench/data/simulations/airline_agent_context_v1.0.1.json \
  --output training/tau2_rollout_sft/generated/tau2_airline_train_success_sft.jsonl
```

The conversion is intentionally strict:

1. It accepts only the 30 IDs in Tau2 v1.0.1's airline train split. A test ID
   or unknown ID aborts the run.
2. It keeps only reward-1 trajectories ending with `user_stop` or `agent_stop`.
3. It rejects trajectories with tool errors or non-empty reasoning/thinking.
4. It creates one prefix/answer row for every genuine agent turn after a user
   or tool message; the canned opening greeting is history, not a target.
5. It includes the exact exported tool schemas and system prompt.
6. It tokenizes the complete row with the Qwen3 tokenizer and discards rows
   above 16,384 tokens.
7. It never copies `tasks`, hidden user-simulator instructions, reference
   actions, evaluation criteria, or raw provider responses.
8. It writes a sibling `.manifest.json` containing counts and the output
   SHA-256.

`--skip-token-count` exists only for offline structural tests and must not be
used for a production dataset.

If you previously converted Tau2 rollouts using Transformers 5 and the old
converter (for example, `metadata.token_count` is always `2`), rerun conversion
to a **new output filename** and regenerate the step-matched AReaL sample from
that corrected reference. Do not merely edit the recorded token counts: some
rows may need to be removed. Existing datasets are not rewritten automatically.

## 5. Build the step-matched AReaL continuation control

After the Tau2 SFT JSONL exists, uniformly sample the same number of rows from
the cleaned AReaL JSONL without replacement:

```bash
python training/tau2_rollout_sft/sample_areal_to_match_tau2.py \
  --tau2-reference training/tau2_rollout_sft/generated/tau2_airline_train_success_sft.jsonl \
  --output training/tau2_rollout_sft/generated/areal_random_matched_to_tau2_rows.jsonl \
  --seed 42
```

The sibling manifest records both input hashes, the seed, selected row count,
and output hash. This is a row-count/optimizer-step control, not a token-count
control: AReaL and Tau2 sequences can have different length distributions.

To make the second-stage optimizer step count identical, both continuation
runs below use the same starting checkpoint, sampled row count, `val_ratio=0`,
epochs, global batch size, learning rate, and all other optimizer settings.

## 6. Run the four SFT ablations

All commands use the LoRA-only SFT entry point (rank 64, alpha 128, LR 1e-4).
Use the dedicated pinned SFT environment in [the SFT README](../qwen3_4b_sft/README.md).
Keep work and checkpoint
directories separate so prepared Parquet and optimizer state cannot cross
experiments.

### AReaL only

```bash
python training/qwen3_4b_sft/train_qwen3_4b_verl_sft.py \
  --work-dir training/qwen3_4b_sft/work/areal_only \
  --output-dir training/qwen3_4b_sft/checkpoints/areal_only \
  --experiment-name areal-only \
  --resume-mode disable \
  --lora-rank 64 --lora-alpha 128 --learning-rate 1e-4 \
  --global-batch-size 8 --optimizer-offload \
  --num-gpus 1 --ulysses-size 1
```

### Tau2 train only

```bash
python training/qwen3_4b_sft/train_qwen3_4b_verl_sft.py \
  --data-jsonl training/tau2_rollout_sft/generated/tau2_airline_train_success_sft.jsonl \
  --work-dir training/qwen3_4b_sft/work/tau2_only \
  --output-dir training/qwen3_4b_sft/checkpoints/tau2_only \
  --experiment-name tau2-only \
  --resume-mode disable \
  --lora-rank 64 --lora-alpha 128 --learning-rate 1e-4 \
  --global-batch-size 8 --optimizer-offload \
  --num-gpus 1 --ulysses-size 1
```

### AReaL then Tau2 train

First finish AReaL-only LoRA SFT, then export and merge its selected checkpoint:

```bash
python agentic_rl/scripts/export_verl_lora.py \
  --stage sft --verl-root /root/verl-sft-v071 \
  --local-dir training/qwen3_4b_sft/checkpoints/areal_only/global_step_N \
  --target-dir /root/models/areal_sft_export
python agentic_rl/scripts/merge_sft_lora.py \
  --sft-adapter /root/models/areal_sft_export/lora_adapter \
  --output /root/models/areal_sft_merged
python agentic_rl/scripts/verify_adapter_equivalence.py \
  --adapter /root/models/areal_sft_export/lora_adapter \
  --merged-model /root/models/areal_sft_merged
```

Replace `global_step_N` with the actual selected checkpoint. Its `huggingface/`
subdirectory is metadata only, not merged weights. Both second-stage branches
must start from `/root/models/areal_sft_merged`, with a **fresh LoRA and optimizer**.
This is not `resume_path` of the first stage:

```bash
python training/qwen3_4b_sft/train_qwen3_4b_verl_sft.py \
  --model /root/models/areal_sft_merged \
  --data-jsonl training/tau2_rollout_sft/generated/tau2_airline_train_success_sft.jsonl \
  --work-dir training/qwen3_4b_sft/work/areal_then_tau2 \
  --output-dir training/qwen3_4b_sft/checkpoints/areal_then_tau2 \
  --experiment-name areal-then-tau2 \
  --val-ratio 0 \
  --epochs 2 \
  --global-batch-size 8 \
  --lora-rank 64 --lora-alpha 128 --learning-rate 1e-4 --optimizer-offload \
  --seed 42 \
  --resume-mode disable \
  --num-gpus 1 --ulysses-size 1
```

### AReaL then step-matched random AReaL

Use the exact same final AReaL-only HF checkpoint and second-stage training
settings as the preceding AReaL-then-Tau2 command:

```bash
python training/qwen3_4b_sft/train_qwen3_4b_verl_sft.py \
  --model /root/models/areal_sft_merged \
  --data-jsonl training/tau2_rollout_sft/generated/areal_random_matched_to_tau2_rows.jsonl \
  --work-dir training/qwen3_4b_sft/work/areal_then_areal_matched \
  --output-dir training/qwen3_4b_sft/checkpoints/areal_then_areal_matched \
  --experiment-name areal-then-areal-matched \
  --val-ratio 0 \
  --epochs 2 \
  --global-batch-size 8 \
  --lora-rank 64 --lora-alpha 128 --learning-rate 1e-4 --optimizer-offload \
  --seed 42 \
  --resume-mode disable \
  --num-gpus 1 --ulysses-size 1
```

This control measures the gain from an equal amount of continued SFT/replay.
The difference from AReaL-then-Tau2 is not explained merely by more optimizer
steps. Sequence/answer lengths, tool-schema availability and data distributions
still differ; this is not a token-matched or schema-matched control.
The matched branches use `2 * floor(reference_rows / 8)` updates. If there are
fewer than eight training rows, reduce the global batch identically for both.
For every final branch, export/merge its own LoRA before the evaluation below;
do not evaluate the unmerged first-stage base by accident.

## 7. Evaluate each model with four test trials

Serve one trained HF model at a time through vLLM. Qwen3's non-thinking tool
format uses the Hermes tool parser:

```bash
vllm serve /root/agentic-rl/PATH_TO_FINAL_HF_MODEL \
  --served-model-name qwen3-4b-sft \
  --max-model-len 16384 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

In another shell:

```bash
export OPENAI_API_BASE=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
export DEEPSEEK_API_KEY='your-key'

python training/tau2_rollout_sft/run_tau2_deepseek.py \
  --tau2-dir /root/tau2-bench \
  --split test \
  --save-name eval_areal_only_test_4trials \
  --num-trials 4 \
  --agent-model openai/qwen3-4b-sft \
  --agent-llm-args '{"temperature":0.0}' \
  --user-model deepseek/deepseek-v4-pro \
  --user-llm-args '{"temperature":0.0,"extra_body":{"thinking":{"type":"disabled"}}}'
```

Repeat with a unique `--save-name` for each of the four checkpoints. Do not use
`base`; evaluation must use `test`.

## 8. Report only pass^1 and pass^4

```bash
python training/tau2_rollout_sft/report_pass1_pass4.py \
  /root/tau2-bench/data/simulations/eval_areal_only_test_4trials/results.json \
  --output-json results/areal_only_pass1_pass4.json \
  --output-md results/areal_only_pass1_pass4.md
```

The reporter reproduces Tau2's official estimator
`C(successes, k) / C(trials, k)` and averages it across all 20 official test tasks.
Before writing metrics it requires the pinned Tau2 commit, airline domain,
the exact test-task list, and every trial declared by `info.num_trials` (at
least four). Trial numbers are zero-based. Simulation IDs must be unique, and
each task/trial may have only one usable result with a finite official reward.

The reporter shares the pinned test IDs and success predicate with the main RL
reporter in `agentic_rl/src/tau2_agentic_rl/pass_metrics.py`. Run it from a complete
repository checkout; it still needs only Python's standard library, not an
installed RL package or GPU dependencies. Rewards must be numeric, non-boolean,
finite and in `[0,1]`. Success uses Tau2's inclusive interval
`1-1e-6 <= reward <= 1+1e-6`, including the exact lower boundary `0.999999`.
Older standalone reports containing that boundary reward should be regenerated
from the existing trajectories; no new rollout or model training is needed.

Official `infrastructure_error` attempts do not count as model failures or as
usable samples. Their missing slots must be completed before reporting; a retry
may fill the same slot with a different simulation ID. Missing whole tasks,
duplicate usable trials, train IDs, partial runs and unscored records are
rejected, without writing final metrics. Both monolithic JSON and the official
`results.json` + `simulations/` directory format are supported. Final JSON and
Markdown still contain only `pass^1` and `pass^4`.

## Offline regression checks

The converter counts tool schemas with the same recursive key ordering as SFT's
prepared `tools_json`. Qwen renders keys in insertion order, so counting raw
schemas could incorrectly reject a sample at the 16,384-token boundary even
when the training input fits. CPU regressions cover that exact boundary; the
real SFT contract job also compares converter length against the actual
Parquet-backed dataset input. This does not rewrite existing datasets: rerun
conversion from the original rollout results to recover previously rejected rows.

The normal `agentic_rl` CPU suite includes the standalone converter and reporter
regressions. With its test dependencies installed, run from `agentic_rl/`:

```bash
export QWEN_TOKENIZER_PATH=/path/to/local/qwen3-4b-instruct-2507-tokenizer
export HF_HUB_OFFLINE=1
python -m pytest tests/test_process_dedup.py tests/test_sft_pipeline_regressions.py
```

The two Transformers CI jobs run these checks with the pinned real Qwen
tokenizer (no model weights, GPU training, user-simulator API or Judge calls).
If `QWEN_TOKENIZER_PATH` is unset, real-tokenizer tests are skipped locally.

The full local offline check on 2026-09-08 used Python 3.13.3 on Windows and
both Transformers 4.57.1 and 5.10.4. Each environment returned **230 passed,
1 skipped** using:

```bash
python -B -m pytest -p no:cacheprovider tests ../training/tau2_rollout_sft/test_pipeline.py -ra
```

The skip is the legacy `RUN_QWEN_INTEGRATION` opt-in test; the new local-tokenizer
tests did run. Ruff checks of the core code/new regressions and syntax/undefined-
name checks of the standalone utilities passed, as did `git diff --check`.
No existing dataset, model weights or live API rollout was changed by validation.

## Leakage boundary

This pipeline prevents direct train/test task-ID mixing and never copies hidden
test instructions. Tau2's train and test tasks still share the same airline
environment/database, so a train trajectory can in principle observe an entity
that also appears in a test task. That is an environment-state-overlap risk,
not direct test-task reproduction; disclose it when reporting the ablation.

## Upstream references

- [Official Tau2 repository](https://github.com/sierra-research/tau2-bench)
- [DeepSeek thinking-mode controls](https://api-docs.deepseek.com/guides/thinking_mode/)
- [Qwen function calling](https://qwen.readthedocs.io/en/stable/framework/function_call.html)
- [vLLM tool-calling server options](https://docs.vllm.ai/en/stable/features/tool_calling/)

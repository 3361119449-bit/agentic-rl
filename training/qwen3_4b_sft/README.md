# Qwen3-4B non-thinking LoRA SFT with verl

Use this training entry point (LoRA only; rank 0 is rejected):

```text
train_qwen3_4b_verl_sft.py
```

It is designed for `verl v0.7.1` and defaults to
`Qwen/Qwen3-4B-Instruct-2507`, Qwen's 4B instruction model that supports only
non-thinking output.

The required veRL v0.7.1 commit is
`bec9ef74768dd201881cd4e54cd0385e87caae27`, with Transformers **4.57.1**.
The actual imported checkout must match and have no tracked modifications.
`lora_sft_runtime.py` adapts the pinned trainer's loaders/checkpoint handling;
`agentic_rl/src/tau2_agentic_rl/sft_contract.py` implements the stdlib preflight.

## What the script does

1. Verifies that the JSONL contains no `thinking`, `reasoning`,
   `reasoning_content`, or `<think>` material.
2. Converts JSONL to nested Parquet without changing the source file.
3. Appends each row's `answer` to its `messages` history.
4. Splits by complete `source_dialog_id`, so sub-trajectories from one dialog
   never cross the train/validation boundary.
5. Uses the full Qwen chat template, including correct grouping of consecutive
   tool responses and optional per-row OpenAI tool schemas.
6. Masks the loss over the complete history and supervises only the final
   appended `answer`.
7. Launches verl's FSDP SFT trainer with a hard 16,384-token limit and no silent
   truncation.

The source data contains 10,649 prefix/answer samples from 909 dialogs. The
default 2% validation split is deterministic by dialog ID. Set
`--val-ratio 0` if every dialog must be used for training.

## AutoDL setup

First clone this repository and fetch the LFS dataset:

```bash
git lfs install
git clone https://github.com/3361119449-bit/agentic-rl.git
cd agentic-rl
git lfs pull
```

Install verl from its tested release. An official verl container is preferable
when available because CUDA, PyTorch, FlashAttention, and NCCL versions must be
compatible.

```bash
git clone https://github.com/verl-project/verl.git ~/verl
git -C ~/verl checkout bec9ef74768dd201881cd4e54cd0385e87caae27
python -m pip install -e ~/verl "transformers==4.57.1" "torchdata==0.11.0" "peft==0.17.1"
python -m pip install -e './agentic_rl[data,test]'
python -m pip check
```

Prepare the Parquet files without starting a GPU job:

```bash
python training/qwen3_4b_sft/train_qwen3_4b_verl_sft.py --prepare-only
```

## LoRA training

Defaults: rank 64, alpha 128, global batch 8, LR `1e-4`, two epochs.
Full-parameter training is not supported by this entry point.

For one GPU, start with LoRA and optimizer offload. A 48 GB or larger GPU is a
safer choice for 16K samples; 24 GB may still be insufficient depending on the
installed kernels.

```bash
python training/qwen3_4b_sft/train_qwen3_4b_verl_sft.py \
  --num-gpus 1 \
  --global-batch-size 8 \
  --lora-rank 64 \
  --lora-alpha 128 \
  --learning-rate 1e-4 --epochs 2 --seed 42 --val-ratio 0.02 \
  --optimizer-offload \
  --experiment-name qwen3-4b-airline-lora \
  --run-name qwen3-4b-airline-lora-r64-lr1e-4-seed42
```

LoRA automatically uses a default learning rate of `1e-4`. Use
`--learning-rate` to override it. Use Python 3.12 and a separate environment
from RL's veRL v0.9.0. These install instructions do not verify GPU kernels
or guarantee that 16K training fits in VRAM.

Every run uses `training/qwen3_4b_sft/runs/<run-name>/checkpoints` and
`resume_mode=disable` by default. The generated default run name includes the
training mode, learning rate, epochs, and seed. Reusing a non-empty output
directory is rejected. Resume only an intentionally selected run:

```bash
python training/qwen3_4b_sft/train_qwen3_4b_verl_sft.py \
  --num-gpus 1 --global-batch-size 8 \
  --lora-rank 64 --lora-alpha 128 --optimizer-offload \
  --learning-rate 1e-4 --epochs 2 --seed 42 --val-ratio 0.02 \
  --run-name qwen3-4b-airline-lora-r64-lr1e-4-seed42 \
  --resume-mode resume_path \
  --resume-from-path /path/to/global_step_N
```

Repeat the original data/work directory, rank/alpha, batch size, learning rate,
offload, epochs, seed and topology when resuming. A run name does **not** recover
CLI settings automatically. Replace `global_step_N` with a completed first-epoch
checkpoint; an already-finished run is rejected.

Resume requires `sft_run_identity.json` and the atomic
`sft_checkpoint_complete.json` inside the selected checkpoint. The identity binds
base weights/tokenizer, source and Parquet hashes, LoRA/optimizer settings,
parallel topology, package versions and SFT code. All model/optimizer/extra-state
shards and `data_<DP rank>.pt` files must exist. Small loader/metadata files are
hashed; large state shards are checked for presence/size, not full corruption.
Do not load untrusted PyTorch checkpoints.

Only completed-epoch checkpoints are supported. Resume restores the model/LoRA,
optimizer, LR scheduler and RNG, then starts the next epoch's seeded sampler.
It does not restore the exhausted previous-epoch iterator (which can skip the
next epoch). Legacy/partial checkpoints without the new manifests cannot resume;
do not fabricate manifests. If later checkpoint directories exist, resume into
a **new** `--output-dir` to avoid overwriting them.

Changing to Tau2 data is a **new SFT stage**, not a resume: first merge the AReaL
LoRA, then use the merged base with a fresh LoRA/optimizer and output directory.

## Export and merge a LoRA checkpoint

The veRL/FSDP checkpoint is not itself a PEFT adapter. Export the selected SFT
`global_step_N` with the pinned veRL v0.7.1 merger:

Training now saves `[model,optimizer,extra]`, not a full HF model.
`global_step_N/huggingface/` contains config/tokenizer metadata, **not** ready-to-
evaluate weights. Use the merged model below for evaluation and the next stage.

```bash
python agentic_rl/scripts/export_verl_lora.py \
  --stage sft --verl-root ~/verl \
  --local-dir training/qwen3_4b_sft/runs/RUN_NAME/checkpoints/global_step_N \
  --target-dir /root/models/qwen3_4b_sft_export

python agentic_rl/scripts/merge_sft_lora.py \
  --sft-adapter /root/models/qwen3_4b_sft_export/lora_adapter \
  --output /root/models/qwen3_4b_airline_sft_merged

python agentic_rl/scripts/verify_adapter_equivalence.py \
  --adapter /root/models/qwen3_4b_sft_export/lora_adapter \
  --merged-model /root/models/qwen3_4b_airline_sft_merged
```

The second command refuses an incomplete adapter directory and verifies the
merged model/tokenizer output. The final GPU check compares adapter and merged
model logits on exactly the same fixed input and fails outside its tolerance.

First install the shared helper package in this SFT environment with
`python -m pip install -e ./agentic_rl`. Training now saves a content-hashed
`base_model_identity.json` beside the checkpoints; export copies it into the
adapter. Merge defaults to that training snapshot and rejects changed weights,
tokenizer or template. A relocated identical snapshot may be supplied with
`--base-model /local/snapshot`. Missing historical manifests are not reconstructed
from today's Hub main; pass the original `--base-identity` when exporting moved
checkpoints. Identity checks and GPU numerical equivalence are separate checks.

## Useful checks and overrides

Print the final verl command without launching training:

```bash
python training/qwen3_4b_sft/train_qwen3_4b_verl_sft.py --dry-run
```

Rebuild Parquet after changing preprocessing options:

```bash
python training/qwen3_4b_sft/train_qwen3_4b_verl_sft.py \
  --prepare-only --force-prepare
```

The preparation cache is reusable only when the resolved source path and
SHA-256, split parameters, model/tokenizer revision, chat-template SHA-256,
Transformers version, context limit, and normalization version all match the
manifest. Equal file size alone is never accepted.

`--model-revision` now resolves one local Hugging Face snapshot for both the
preprocessing tokenizer and veRL `model.path`, so a pinned tokenizer cannot
silently train against a newer `main`. Normal training downloads the snapshot's
weights; `--prepare-only` / `--dry-run` download only tokenizer/config metadata.
For a local `--model` directory, that directory is used directly. When exporting,
merging or checking the adapter later, use the same resolved base-model snapshot
path recorded in the preparation manifest, not a moving Hub `main`.

Raw Hydra overrides can be appended when needed:

```bash
python training/qwen3_4b_sft/train_qwen3_4b_verl_sft.py \
  --extra-config engine.use_torch_compile=false
```

Generated Parquet files and checkpoints are intentionally ignored by Git.

Only `engine.use_torch_compile=true/false` raw overrides are accepted. Critical
model, data, loss, checkpoint and resume settings cannot bypass the explicit CLI.

After upgrading from preparation version 4, use `--force-prepare` once or a new
work directory. Cache reuse additionally verifies the Parquet file hashes.
The default Qwen model revision is pinned to
`cdbee75f17c01a7cc42f958dc650907174af0554`; local model directories remain supported.

## Validation, seed and test boundaries

`--val-ratio 0` safely disables validation (`test_freq=-1`). Empty validation
splits are also disabled. Training needs at least one full global batch and uses
`drop_last=True`: steps = `floor(train_rows / global_batch_size) * epochs`.

Validation is complete, batch size 1, with no discarded tail. Every DP rank
evaluates the same small validation set, avoiding padding duplicates and unequal
collective counts even when there are fewer examples than GPUs. Reported loss
is the mean of each example's answer-token NLL, not token-weighted corpus
perplexity or Tau2 pass metrics. The tradeoff is redundant validation across DP.

The seed controls splitting, the training sampler, independent loader generators
and LoRA/model initialization; it does not promise bitwise GPU determinism.
CPU regressions and pinned-source contract checks are not a substitute for an
AutoDL LoRA train → epoch-boundary resume → export → merge → GPU-equivalence
smoke test. No full-parameter training acceptance is claimed.

Local verification on 2026-09-06: 141 standard CPU regressions and 8 isolated
pinned-source loader/fit/resume checks passed. The rollout conversion suite had
3 passes / 1 optional tokenizer test skipped. All 10,649 bundled rows were
reprocessed and checked with the real Qwen3-4B-Instruct-2507 tokenizer and the
custom dataset getter: max 16,310 tokens, no overlength/empty-target/prefix errors,
and no train/validation dialog overlap. This getter check did not instantiate
the full veRL engine.

The new `real-sft-lora-contract` CI job separately checks installed veRL imports,
Hydra configuration and real parent-dataset/Parquet integration. Those 3 full-
stack checks were not run locally (skipped in isolated mode); the new CI job and
GPU acceptance have not yet been executed. To run the full CPU contract suite
in the installed SFT environment, from `agentic_rl/`:

```bash
RUN_REAL_SFT_CONTRACT=1 QWEN_TOKENIZER_PATH=/path/to/local/qwen-tokenizer \
  python -m pytest sft_contract_tests -v
```

## Important data note

The published AReaL JSONL contains message history and tool calls, but no
top-level tool-schema list. This script preserves that data exactly. During Tau2
evaluation, the agent runtime should still provide the current airline tool
schemas to the model; do not hard-code test environment state into SFT data.

JSONL produced by `training/tau2_rollout_sft/convert_tau2_results_to_sft.py`
does contain a top-level `tools` list. The preparation step stores it in
Parquet, and the custom dataset passes it to Qwen's chat template for every
row.

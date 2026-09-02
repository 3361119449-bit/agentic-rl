# Qwen3-4B airline SFT with verl

This directory contains one training entry point:

```text
train_qwen3_4b_verl_sft.py
```

It is designed for `verl v0.7.1` and defaults to
`Qwen/Qwen3-4B-Instruct-2507`, Qwen's 4B instruction model that supports only
non-thinking output.

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
git clone --branch v0.7.1 https://github.com/verl-project/verl.git ~/verl
python -m pip install -e ~/verl
python -m pip install "transformers>=4.51.0" pyarrow
```

Prepare the Parquet files without starting a GPU job:

```bash
python training/qwen3_4b_sft/train_qwen3_4b_verl_sft.py --prepare-only
```

## Full-parameter SFT

Full-parameter training is the default. This example uses eight GPUs and
two-way Ulysses sequence parallelism for the 16K trajectories:

```bash
python training/qwen3_4b_sft/train_qwen3_4b_verl_sft.py \
  --num-gpus 8 \
  --ulysses-size 2 \
  --global-batch-size 32 \
  --micro-batch-size 1
```

The default learning rate is `2e-5`, training lasts two epochs, validation and
checkpointing run after every epoch, and the output includes an HF-format model.

## Lower-memory LoRA run

For one GPU, start with LoRA and optimizer offload. A 48 GB or larger GPU is a
safer choice for 16K samples; 24 GB may still be insufficient depending on the
installed kernels.

```bash
python training/qwen3_4b_sft/train_qwen3_4b_verl_sft.py \
  --num-gpus 1 \
  --global-batch-size 8 \
  --lora-rank 64 \
  --lora-alpha 128 \
  --optimizer-offload \
  --experiment-name qwen3-4b-airline-lora
```

LoRA automatically uses a default learning rate of `1e-4`. Use
`--learning-rate` to override either default.

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

Raw Hydra overrides can be appended when needed:

```bash
python training/qwen3_4b_sft/train_qwen3_4b_verl_sft.py \
  --extra-config engine.use_torch_compile=false
```

Generated Parquet files and checkpoints are intentionally ignored by Git.

## Important data note

The published AReaL JSONL contains message history and tool calls, but no
top-level tool-schema list. This script preserves that data exactly. During Tau2
evaluation, the agent runtime should still provide the current airline tool
schemas to the model; do not hard-code test environment state into SFT data.

JSONL produced by `training/tau2_rollout_sft/convert_tau2_results_to_sft.py`
does contain a top-level `tools` list. The preparation step stores it in
Parquet, and the custom dataset passes it to Qwen's chat template for every
row.

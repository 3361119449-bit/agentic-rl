# agentic-rl

This repository contains the cleaned AReaL Tau2 airline SFT dataset and its reproducibility artifacts.

## Dataset

Path: `datasets/tau2_airline_sft_strict_cleaned/`

```text
datasets/tau2_airline_sft_strict_cleaned/
├── data/          # Cleaned JSONL dataset
├── code/          # Reproducible filtering script
├── manifest/      # Removed source-dialog IDs and cleanup tiers
├── docs/          # Cleaning procedure
└── verification/  # Counts, checksums, and validation results
```

The cleaned dataset contains:

- 10,649 SFT rows
- 909 source dialogs
- no `thinking` or `reasoning` fields
- no `<think>` tags
- only samples already verified to fit the Qwen3 4B 16k context limit

The strict leakage cleanup removed 90 complete source dialogs (1,198 prefix rows). Post-cleaning audit found zero exact Tau2 test reference queries, zero same-day test-target flight exposures, and zero retained high-overlap templates for tests 13, 32, and 48.

Dataset SHA-256:

```text
5535171a7b6f271d282966ffcb5d5b1a7a5581bc2aa7226667624d9e20ed4cb0
```

See [`CLEANING_PROCESS.md`](datasets/tau2_airline_sft_strict_cleaned/docs/CLEANING_PROCESS.md) for the full procedure.

## Download with Git LFS

The JSONL file is stored with Git LFS because it exceeds GitHub's normal single-file size limit.

```bash
git lfs install
git clone https://github.com/3361119449-bit/agentic-rl.git
cd agentic-rl
git lfs pull
```

## Train Qwen3-4B with verl

The repository includes a verl SFT entry point for
`Qwen/Qwen3-4B-Instruct-2507`. It prepares the JSONL as Parquet, splits by full
source dialog, and supervises only each row's final `answer` rather than the
assistant messages in its history.

See [`training/qwen3_4b_sft/README.md`](training/qwen3_4b_sft/README.md) for
AutoDL setup, full-parameter SFT, and lower-memory LoRA commands.

## Tau2 train rollout SFT and evaluation

The repository also includes a reproducible pipeline that:

- rolls out official Tau2 airline train tasks with DeepSeek V4 Flash as the
  agent and DeepSeek V4 Pro as the user simulator;
- disables and rejects reasoning/thinking content;
- converts only successful train trajectories to 16K Qwen3 SFT rows;
- runs AReaL-only, Tau2-only, AReaL-then-Tau2, and a step-matched
  AReaL-then-random-AReaL continuation control; and
- evaluates the official test split while reporting only `pass^1` and
  `pass^4`.

See
[`training/tau2_rollout_sft/README.md`](training/tau2_rollout_sft/README.md).

No license is asserted here for upstream datasets or repositories. Users should follow the terms of the original data sources.

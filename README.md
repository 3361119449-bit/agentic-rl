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

No license is asserted here for upstream datasets or repositories. Users should follow the terms of the original data sources.

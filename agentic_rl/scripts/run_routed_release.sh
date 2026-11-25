#!/usr/bin/env bash
# Entry point for the side-by-side AutoDL release. Use --run for training.
set -euo pipefail
project="$(cd -- "$(dirname -- "$0")/.." && pwd -P)"
base=/root/autodl-tmp
python="$base/venvs/airline-rl/bin/python"
set -a
set +u
source "$base/agentic-rl/agentic_rl/.env"
set -u
set +a
export MERGED_SFT_MODEL=/root/sft_eval_v4_20261106_merged
export TAU2_ROOT="$base/tau2-bench-data" TAU2_DATA_DIR="$base/tau2-bench-data/data"
export VERL_ROOT="$(dirname "$project")/verl"
export PYTHONPATH="$project/src:$project:$TAU2_ROOT/src:$VERL_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export PYTHONDONTWRITEBYTECODE=1 LITELLM_LOCAL_MODEL_COST_MAP=True
mode=(--dry-run)
if [[ ${1:-} == --run ]]; then
    shift
    mode=()
fi
cd "$project"
exec "$python" -u scripts/train_airline_grpo.py --stage full_train \
    --config configs/rl/airline_procredit_v4_train30_c32_routed.yaml "${mode[@]}" "$@"

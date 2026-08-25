#!/bin/bash
set -euo pipefail

# --- Environment setup (must be self-contained for phd run) ---
export HOME=${HOME:-/home/sr5/sairaj.loke}

# Activate venv
source $HOME/other/venv312_new/bin/activate

# Set LD_LIBRARY_PATH (same as setup_new_env_vars.sh, no cu13)
export LD_LIBRARY_PATH=\
$HOME/other/shim312_new:\
$VIRTUAL_ENV/lib/python3.12/site-packages/av.libs:\
$HOME/other/uvexpt/python/lib:\
${LD_LIBRARY_PATH:-}

# Working directory — must be ViTacFormer for relative paths to work
cd $HOME/other/ori/ori/ViTacFormer

# --- Dataset location (REQUIRED: configs.py reads DATASET_ROOT and now fails
#     loudly if it is unset, instead of raising Path(None) -> TypeError) ---
export DATASET_ROOT="${DATASET_ROOT:-$HOME/other/new_data/larger_data}"
echo "DATASET_ROOT     : $DATASET_ROOT"
echo "MAX_EPISODES     : ${MAX_EPISODES:-0} (0=all)"
echo "VAL_EPISODES     : ${VAL_EPISODES:-0,1}"
echo "VAL_EVERY_N_EPOCHS: ${VAL_EVERY_N_EPOCHS:-10}"
echo "BACKBONE_WEIGHTS : ${BACKBONE_WEIGHTS:-<torch hub ImageNet>}"
echo "USE_NORMALIZATION: ${USE_NORMALIZATION:-0}"
echo "MIXED_PRECISION  : ${MIXED_PRECISION:-bf16}"

# --- Run the multi-GPU training ---
# Usage: bash run_ori_job.sh [NUM_GPUS]
# Default: 4 GPUs
NUM_GPUS="${1:-4}"

bash origami_train_multigpu.sh "$NUM_GPUS"



# phd run -ng 4 -p shr_gpu -GR A100 -l %J.log sh run_ori_job.sh 4



# phd run -ng 8 -p shr_gpu -GR H100 -l %J.log sh run_ori_job.sh 8


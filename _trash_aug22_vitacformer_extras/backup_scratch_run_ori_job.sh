#!/bin/bash
set -euo pipefail

# --- Raise file descriptor limit (prevents "too many open files" with many DataLoader workers) ---
ulimit -n 65536

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

# ============================================================================
# --- Configurable dataset source (env var override) ---
# ============================================================================
# Default: larger_data. Override with DATASET_SRC env var.
DATASET_SRC="${DATASET_SRC:-$HOME/other/new_data/larger_data}"

# ============================================================================
# --- Scratch storage: copy dataset to local NVMe to eliminate NFS I/O spikes ---
# ============================================================================
DATASET_DST=""
JOB_ID="${B2_JOB_ID:-$$}"

# Detect scratch path on compute node (only exists on GPU nodes, not login)
for scratch_dir in /scratch_h4 /scratch_h8 /scratch_a4 /scratch_a8; do
    if [ -d "$scratch_dir" ]; then
        DATASET_DST="${scratch_dir}/${USER}/${JOB_ID}/dataset"
        break
    fi
done

if [ -n "$DATASET_DST" ]; then
    echo "[Scratch] Found scratch at $(dirname $DATASET_DST)"
    echo "[Scratch] Copying dataset from $DATASET_SRC to $DATASET_DST ..."
    mkdir -p "$DATASET_DST"
    cp -r "$DATASET_SRC"/* "$DATASET_DST/"
    export DATASET_ROOT="$DATASET_DST"
    echo "[Scratch] Dataset copied. Training from local NVMe."

    # Cleanup scratch on exit (normal or killed)
    cleanup_scratch() {
        echo "[Scratch] Cleaning up $DATASET_DST ..."
        rm -rf "$DATASET_DST"
    }
    trap cleanup_scratch EXIT
else
    echo "[Scratch] No scratch dir found. Training from NFS (home storage)."
    export DATASET_ROOT="$DATASET_SRC"
fi
# ============================================================================

# --- Print config summary ---
echo "============================================"
echo "MIXED_PRECISION : ${MIXED_PRECISION:-bf16}"
echo "MAX_EPISODES    : ${MAX_EPISODES:-0} (0=all)"
echo "USE_NORMALIZATION: ${USE_NORMALIZATION:-0}"
echo "DATASET_SRC     : $DATASET_SRC"
echo "============================================"

# --- Run the multi-GPU training ---
# Usage: bash run_ori_job.sh [NUM_GPUS]
# Default: 4 GPUs
NUM_GPUS="${1:-4}"

bash origami_train_multigpu.sh "$NUM_GPUS"

# ============================================================================
# Example submission commands for different variants:
# ============================================================================
# Variant 1: bf16, 50 episodes, no normalization
#   MIXED_PRECISION=bf16 MAX_EPISODES=50 USE_NORMALIZATION=0 \
#   phd run -ng 4 -p shr_gpu -GR H100 -l %J.log sh run_ori_job.sh 4
#
# Variant 2: fp32, 50 episodes, no normalization
#   MIXED_PRECISION=fp32 MAX_EPISODES=50 USE_NORMALIZATION=0 \
#   phd run -ng 4 -p shr_gpu -GR H100 -l %J.log sh run_ori_job.sh 4
#
# Variant 3: fp16, 50 episodes, no normalization
#   MIXED_PRECISION=fp16 MAX_EPISODES=50 USE_NORMALIZATION=0 \
#   phd run -ng 4 -p shr_gpu -GR H100 -l %J.log sh run_ori_job.sh 4
#
# Variant 4: bf16, 50 episodes, WITH normalization
#   MIXED_PRECISION=bf16 MAX_EPISODES=50 USE_NORMALIZATION=1 \
#   phd run -ng 4 -p shr_gpu -GR H100 -l %J.log sh run_ori_job.sh 4
# ============================================================================


# cd /home/sr5/sairaj.loke/other && MIXED_PRECISION=bf16 MAX_EPISODES=50 USE_NORMALIZATION=0 phd run -ng 4 -p shr_gpu -GR H100 -l %J.log sh run_ori_job.sh 4
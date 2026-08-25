#!/bin/bash
# Local CPU smoke test: exercises dataset -> convert_batch -> policy -> loss ->
# backward -> validation -> checkpoint on a real season, in a couple of minutes.
# Not a training run; --max_train_steps/--max_val_steps cut every loop short.
#
#   bash smoke_test.sh [DATASET_ROOT]
set -euo pipefail

VENV="$(dirname "$0")/.venv"
PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "no venv at $VENV -- see req_smoke_cpu.txt for setup"; exit 1; }

export DATASET_ROOT="${1:-/media/sai/CRUZER_BLA/ori/dataset/season_POC22061_2026_07_09_16_23_46_train/lerobot3.0}"
export MAX_EPISODES="${MAX_EPISODES:-4}"
export VAL_EPISODES="${VAL_EPISODES:-0,1}"
export VAL_EVERY_N_EPOCHS="${VAL_EVERY_N_EPOCHS:-1}"
export ORI_LOG_LEVEL="${ORI_LOG_LEVEL:-DEBUG}"
export ORI_LOG_LEVELS="${ORI_LOG_LEVELS:-data=TRACE,norm=INFO,model=INFO}"
export CUDA_VISIBLE_DEVICES=""

# DEV-BOX ONLY: miniconda ships libffi.so.8 which shadows the system libffi.so.7
# that libgobject (and hence FFmpeg 4, and hence libtorchcodec) links against.
# Symptom without this: "Could not load libtorchcodec ... undefined symbol:
# ffi_type_uint32, version LIBFFI_BASE_7.0". Not needed on the cluster.
if [ -e /lib/x86_64-linux-gnu/libffi.so.7 ]; then
    export LD_PRELOAD="/lib/x86_64-linux-gnu/libffi.so.7${LD_PRELOAD:+:$LD_PRELOAD}"
fi

OUT="${OUT:-/tmp/ori_smoke_out}"
rm -rf "$OUT"; mkdir -p "$OUT"

echo "DATASET_ROOT = $DATASET_ROOT"
echo "MAX_EPISODES = $MAX_EPISODES   VAL_EPISODES = $VAL_EPISODES"
echo "normalization: ${NORM_FLAG:---disable_normalization}"

"$PY" origami_imitate_episodes.py \
  --task_name fold_plane \
  --expt_name smoke \
  --policy_class ACT --kl_weight 10 --hidden_dim 256 \
  --batch_size 2 --dim_feedforward 512 \
  --ckpt_save_epochs 1 \
  --num_epochs 2 \
  --lr 3e-4 \
  --seed 0 \
  --use_tactile \
  --state_dim 65 \
  --ckpt_dir "$OUT" \
  --tb_log_freq 1 \
  --max_train_steps 3 \
  --max_val_steps 2 \
  ${NORM_FLAG---disable_normalization} \
  "${@:2}"

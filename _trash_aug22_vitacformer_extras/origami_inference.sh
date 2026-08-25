#!/bin/bash
# Origami Inference & Open-Loop Evaluation
#
# Usage:
#   bash origami_inference.sh
#
# Adjust CKPT_DIR, CKPT_FILE, and STATS_FILE below to match your trained model.
#
# Two modes:
#   1) Sample-based eval (default): evaluates on N random sequential samples
#   2) Episode-based eval: runs inference over a single complete episode
#      in non-overlapping chunks of pred_horizon (default 64), saving
#      per-segment predicted joint poses.
#
# To use episode mode, set EPISODE_IDX to the desired episode number.
# Leave EPISODE_IDX empty (or comment it out) for sample-based eval.

CKPT_DIR="ckpt_dir/fold_plane/jul12_unmasked_unnormed_12epis_continued1_20260713_222821_ori_tactile"
CKPT_FILE="policy_epoch_30_loss_0.041.ckpt"
STATS_FILE="normalize.pkl"
OUTPUT_DIR="inference_results/jul12_unmasked_unnormed_12epis_continued1_${CKPT_FILE}"

# ── Episode mode settings ──────────────────────────────────────────
# Set EPISODE_IDX to run over a single complete episode in non-overlapping
# chunks of PRED_HORIZON steps. Leave empty for sample-based eval.
EPISODE_IDX=0
PRED_HORIZON=64

# ── Build command ──────────────────────────────────────────────────
CMD="python3 origami_inference.py \
    --ckpt_path \"${CKPT_DIR}/${CKPT_FILE}\" \
    --stats_path \"${CKPT_DIR}/${STATS_FILE}\" \
    --use_tactile \
    --output_dir \"${OUTPUT_DIR}\" \
    --device cuda"

if [ -n "$EPISODE_IDX" ]; then
    echo "=== Episode mode: episode_idx=${EPISODE_IDX}, pred_horizon=${PRED_HORIZON} ==="
    CMD="${CMD} --episode_idx ${EPISODE_IDX} --pred_horizon ${PRED_HORIZON}"
else
    echo "=== Sample-based mode ==="
    CMD="${CMD} --n_samples 50 --batch_size 1"
fi

eval $CMD

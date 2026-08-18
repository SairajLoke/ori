#!/bin/bash
set -euo pipefail

# Multi-GPU training script using Accelerate
# Usage: ./origami_train_multigpu.sh [num_gpus] [other args...]
# Example: ./origami_train_multigpu.sh 4 --batch_size 128 --num_epochs 1000
#
# Environment variables (all optional):
#   MIXED_PRECISION=bf16|fp16|fp32  (default: bf16)
#   USE_NORMALIZATION=0|1            (default: 0, i.e. --disable_normalization)
#   EXPT_NAME=custom_name            (default: auto-generated from precision+episodes)
#   MAX_EPISODES=50                  (default: 0 = all episodes, read by Python via configs.py)
#
# Note: --load_pretrained_for_newtraining and --resume_path are mutually exclusive:
#   - Use --load_pretrained_for_newtraining to load a pretrained model and train from scratch
#   - Use --resume_path to resume training from a checkpoint (continues from epoch/step)
#   - You cannot use both at the same time
#   - You MUST provide exactly one of them (training from scratch is not allowed)

# Check if the first argument is missing or empty
if [ -z "$1" ]; then
    echo "Error: NUM_GPUS is a required argument."
    echo "Usage: $0 <NUM_GPUS>"
    exit 1
fi

NUM_GPUS=$1
shift

# Validate NUM_GPUS is one of the supported values
if [[ ! "$NUM_GPUS" =~ ^(1|2|4|8)$ ]]; then
    echo "ERROR: NUM_GPUS must be one of: 1, 2, 4, 8. Got: $NUM_GPUS"
    exit 1
fi

# --- Mixed precision config selection ---
# Config files are named uniformly: accelerate_config_<N>gpu_<precision>.yaml
# for every N in {1,2,4,8} and every precision in {bf16,fp16,fp32}. There are no
# special cases and no unsuffixed files, so a request can never silently
# resolve to a config with a different precision than the one you asked for.
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
case "$MIXED_PRECISION" in
    bf16|fp16|fp32) ;;
    *)
        echo "ERROR: MIXED_PRECISION must be one of: bf16, fp16, fp32. Got: $MIXED_PRECISION"
        exit 1
        ;;
esac
ACCELERATE_CONFIG="accelerate_configs/accelerate_config_${NUM_GPUS}gpu_${MIXED_PRECISION}.yaml"

if [ ! -f "$ACCELERATE_CONFIG" ]; then
    echo "ERROR: Accelerate config file not found: $ACCELERATE_CONFIG"
    echo "Available configs:"
    ls -1 accelerate_configs/ | sed 's/^/  /'
    exit 1
fi

# Fail fast if the file's precision does not match what was requested.
case "$MIXED_PRECISION" in
    fp32) EXPECT_MP="'no'" ;;
    *)    EXPECT_MP="'${MIXED_PRECISION}'" ;;
esac
ACTUAL_MP=$(grep -E "^mixed_precision:" "$ACCELERATE_CONFIG" | awk '{print $2}')
if [ "$ACTUAL_MP" != "$EXPECT_MP" ]; then
    echo "ERROR: $ACCELERATE_CONFIG declares mixed_precision=$ACTUAL_MP but MIXED_PRECISION=$MIXED_PRECISION expects $EXPECT_MP"
    exit 1
fi

# --- Normalization flag ---
USE_NORMALIZATION="${USE_NORMALIZATION:-0}"
NORM_FLAG=""
if [ "$USE_NORMALIZATION" = "1" ]; then
    NORM_FLAG=""  # normalization enabled = no flag
else
    NORM_FLAG="--disable_normalization"
fi

# --- Experiment name ---
MAX_EPISODES_STR="${MAX_EPISODES:-0}"
if [ "$MAX_EPISODES_STR" = "0" ]; then
    EPISODES_LABEL="allepi"
else
    EPISODES_LABEL="${MAX_EPISODES_STR}epi"
fi
NORM_LABEL="unnorm"
if [ "$USE_NORMALIZATION" = "1" ]; then
    NORM_LABEL="norm"
fi
EXPT_NAME="${EXPT_NAME:-aug17_8xh100_${MIXED_PRECISION}_${EPISODES_LABEL}_${NORM_LABEL}}"

# Build accelerate launch command
ACCELERATE_CMD="accelerate launch --config_file $ACCELERATE_CONFIG --num_processes $NUM_GPUS"

echo "Launching training on $NUM_GPUS GPUs..."
echo "Config file: $ACCELERATE_CONFIG"
echo "Mixed precision: $MIXED_PRECISION"
echo "Normalization: $USE_NORMALIZATION ($NORM_LABEL)"
echo "Max episodes: $MAX_EPISODES_STR"
echo "Experiment name: $EXPT_NAME"
echo "Command: $ACCELERATE_CMD origami_imitate_episodes.py $@"
echo ""

# Increase torchcodec decoder LRU cache (default=100 decoders).
# Unit = number of open video decoders (not bytes). Each entry ~few MB.
# 500 comfortably covers many chunk files across 4 cameras without eviction.
export LEROBOT_VIDEO_DECODER_CACHE_SIZE="${LEROBOT_VIDEO_DECODER_CACHE_SIZE:-500}"

# Build/install local modules
cd dataset/ha_data && pip install -e . && cd ../..
cd detr && pip install -e . && cd ..


$ACCELERATE_CMD origami_imitate_episodes.py \
--task_name fold_plane \
--expt_name  "$EXPT_NAME" \
--policy_class ACT --kl_weight 10 --hidden_dim 512 \
--batch_size 64 --dim_feedforward 3200 \
--ckpt_save_epochs 1 \
--num_epochs 100  \
--lr 3e-4 \
--seed 0 \
--use_tactile \
--state_dim 65 \
--visualize_batch \
--doImageTransforms \
--ckpt_dir ckpt_dir/fold_plane \
--tb_log_freq 10 \
--resume_path "/home/sr5/sairaj.loke/other/ori/ori/ViTacFormer/ckpt_dir/fold_plane/aug14_h100_20260816_202730_ori_tactile/policy_epoch_23_loss_0.034.ckpt" \
$NORM_FLAG




##########################
# --resume_path 

# --load_pretrained_for_newtraining "/home/sr5/sairaj.loke/other/ori/ori/ViTacFormer/ckpt_dir/fold_plane/aug5_continued2_multigpu_fullepi_99epi_unmasked_unnormed_weigthfrom_jul12_unmasked_unnormed_12epis_continued1_d_20260805_001037_ori_tactile/policy_globalstep_2200_loss_0.08123782277107239.ckpt"


# cnode                    status    jobs                                                                                                                               c_avl        c_used    m_avl       m_tot       swp_avl     swp_tot    r1m            r5m    r15m  controller
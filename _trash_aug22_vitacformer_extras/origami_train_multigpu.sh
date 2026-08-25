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
#   VAL_EPISODES=0,1                 held-out episode indices (default 0,1; "" disables validation)
#   VAL_EVERY_N_EPOCHS=10            validation cadence
#   ORI_IMAGE_NORM=0|1               ImageNet mean/std on camera images (default 1)
#   ORI_NORM_DISABLE_KEYS=k1,k2      force these features to identity while the rest normalize
#   ORI_USE_OBS_FPS=0|1              observation.state/tactile history spaced at OBS_FPS, not FPS (default 0)
#   ORI_OBS_FPS=5.0                  the OBS_FPS value used above (default 5.0)
#   ORI_JITTER_HISTORY=0|1           randomize per-step gaps in the observation.state/tactile
#                                    PAST history window per training step, to emulate irregular
#                                    inference cadence (default 0). tactile_next/action targets
#                                    and images are never jittered. See configs.py for details.
#   ORI_JITTER_MAX_GAP_MULT=3.0      max jittered gap, as a multiple of the regular step gap (default 3.0)
#   BACKBONE_WEIGHTS=/path.pth       local ResNet ckpt instead of the torch hub download
#   RESUME_PATH=/path.ckpt           resume (weights + optimizer + scheduler + epoch/step)
#   PRETRAINED_PATH=/path.ckpt       weights only, fresh schedule
#   EXTRA_ARGS="--foo 1 --bar 2"     appended verbatim to the python command line
#   ORI_LOG_LEVEL / ORI_LOG_LEVELS / ORI_LOG_RANKS   see my_utils/ori_logging.py
#
# RESUME_PATH and PRETRAINED_PATH are mutually exclusive; setting neither trains
# from random init. NOTE: the tactile/state delta_timestamps fix changed the
# model's input semantics, so checkpoints predating it are NOT resumable.

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

# --- Checkpoint source (env-driven; empty = fresh random init) ---
# These used to be hardcoded below. They must be settable per-run because the
# tactile/state delta_timestamps fix changed the model's input semantics: any
# checkpoint from before that change is NOT resumable.
#   RESUME_PATH=...      continue a run (epoch, optimizer, scheduler, step)
#   PRETRAINED_PATH=...  take weights only, start a fresh schedule
EXTRA_ARGS="${EXTRA_ARGS:-}"          # appended verbatim to the python command line
RESUME_PATH="${RESUME_PATH:-}"
PRETRAINED_PATH="${PRETRAINED_PATH:-}"
if [ -n "$RESUME_PATH" ] && [ -n "$PRETRAINED_PATH" ]; then
    echo "ERROR: set only one of RESUME_PATH / PRETRAINED_PATH"
    exit 1
fi
CKPT_FLAG=""
if [ -n "$RESUME_PATH" ]; then
    CKPT_FLAG="--resume_path $RESUME_PATH"
elif [ -n "$PRETRAINED_PATH" ]; then
    CKPT_FLAG="--load_pretrained_for_newtraining $PRETRAINED_PATH"
else
    echo "NOTE: neither RESUME_PATH nor PRETRAINED_PATH set -- training from random init."
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
echo "Checkpoint     : ${CKPT_FLAG:-<fresh random init>}"
echo "Image norm     : ${ORI_IMAGE_NORM:-1}"
echo "Norm excl keys : ${ORI_NORM_DISABLE_KEYS:-<none>}"
echo "Extra args     : ${EXTRA_ARGS:-<none>}"
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
--ckpt_save_epochs 5 \
--num_epochs 100  \
--lr 3e-4 \
--seed 0 \
--use_tactile \
--state_dim 65 \
--visualize_batch \
--doImageTransforms \
--ckpt_dir ckpt_dir/fold_plane \
--tb_log_freq 10 \
$CKPT_FLAG \
$EXTRA_ARGS \
$NORM_FLAG




##########################
# --resume_path 

# --load_pretrained_for_newtraining "/home/sr5/sairaj.loke/other/ori/ori/ViTacFormer/ckpt_dir/fold_plane/aug5_continued2_multigpu_fullepi_99epi_unmasked_unnormed_weigthfrom_jul12_unmasked_unnormed_12epis_continued1_d_20260805_001037_ori_tactile/policy_globalstep_2200_loss_0.08123782277107239.ckpt"


# cnode                    status    jobs                                                                                                                               c_avl        c_used    m_avl       m_tot       swp_avl     swp_tot    r1m            r5m    r15m  controller
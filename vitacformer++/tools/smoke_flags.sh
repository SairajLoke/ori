#!/usr/bin/env bash
# Smoke-test the action/observation flags: each alone, then combined.
# Lives in the repo (not /tmp) so it survives a tmp clean.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
VENV=${VENV:-/home/sai/Desktop/ORI/ori/origami-inference-kit-participant/sharpa_north_ces_lite_sdk-main/.venv}
export DATASET_ROOT=${DATASET_ROOT:-/media/sai/CRUZER_BLA/ori/dataset/season_POC22061_2026_07_09_16_23_46_train/lerobot3.0_shortgop15_224}
export ORI_VIDEO_BACKEND=${ORI_VIDEO_BACKEND:-pyav}
export PYTHONPATH="$PWD:$PWD/detr"
export MAX_EPISODES=${MAX_EPISODES:-3} VAL_EPISODES=${VAL_EPISODES:-0} VAL_EVERY_N_EPOCHS=1
OUT=${OUT:-/tmp/flagsmoke}
W="--loss_dim_weight_mode file --action_weights_json action_weights.json"

run () {
  tag=$1; shift
  echo "######## $tag ########"
  "$VENV/bin/python3" origami_imitate_episodes.py \
    --ckpt_dir "$OUT/$tag" --ckpt_save_epochs 99 --expt_name "$tag" \
    --policy_class ACT --task_name fold_plane --batch_size 4 --seed 0 \
    --num_epochs 1 --lr 3e-4 --kl_weight 10 --hidden_dim 512 --dim_feedforward 3200 \
    --use_tactile --state_dim 65 --tb_log_freq 5 --max_train_steps 3 --max_val_steps 1 \
    $W "$@" 2>&1 \
    | grep -E "cameras:|image_crop:|tactile_mode=|action_dim=|constant dims|VAL epoch|Traceback|Error" \
    | grep -viE "userwarning" | head -8
  echo "   exit=${PIPESTATUS[0]}"
}

run crop192       --image_crop 192
run fast_attn     --fast_attn
run combo         --tactile_mode input --cameras head_left,wrist_right,wrist_left \
                  --image_crop 192 --action_dims_mode active --temporal_weight_mode horizon
echo "######## SMOKE DONE ########"

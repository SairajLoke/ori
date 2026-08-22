#!/usr/bin/env bash
# =============================================================================
# REFERENCE ONLY -- not meant to be executed top to bottom.
# Copy the block you want. See EXPERIMENTS.md for the reasoning behind each.
# =============================================================================
exit 0   # guard: stops accidental `bash cmds.sh` from running everything

# -----------------------------------------------------------------------------
# 0. Paths / common env
# -----------------------------------------------------------------------------
export R=$HOME/other/ori/ori/vitacformer++
export SUB=$HOME/other/ori/ori/orivizkar-submission-v000-gpu
export DATASET_SRC=$HOME/other/new_data/YOUR_224_SHORTGOP_ROOT   # <-- edit

export BASE="USE_SCRATCH=1 MIXED_PRECISION=bf16 USE_NORMALIZATION=1 MAX_EPISODES=30"
export PHD="phd run -ng 4 -p shr_gpu -GR H100 -l %J.log sh $R/run_ori_job.sh 4"

# The JSON holds four INDEPENDENT sections (finger/group weights, constant_dims,
# delta_stats). Passing it alone changes NOTHING -- each section is switched on by
# its own flag. So $J is safe to include everywhere and every run below moves
# exactly one variable off the baseline.
export J="--action_weights_json action_weights.json"

# -----------------------------------------------------------------------------
# 1. PREPROCESS -- run once, before any experiment
# -----------------------------------------------------------------------------

# 1a. Action weights from YOUR data. --episodes 30 so the stats come from the
#     SAME episodes you train on (MAX_EPISODES=30 takes the first 30).
cd $R && python tools/compute_action_weights.py \
  --dataset_root $DATASET_SRC --episodes 30 --out action_weights.json
#   expect: thumb > index > middle >> ring ~ pinky
#           constant dims {58, 59, 64:-0.8727}
#           delta spread floored on a handful of 65 dims

# 1b. Is fused attention actually firing, and what does it save?
cd $R && python tools/check_sdpa.py --batch 64 --seq 200 --dtype bfloat16
cd $R && python tools/check_sdpa.py --batch 96 --seq 200 --dtype bfloat16

# 1c. Vendored model source must match vitacformer++ before ANY image build
cd $SUB && ./scripts/sync_vendor.sh --check   # exits 1 if stale
cd $SUB && ./scripts/sync_vendor.sh           # sync it

# -----------------------------------------------------------------------------
# 2. EXPERIMENTS -- ONE variable each, all against E0
# -----------------------------------------------------------------------------

# E0  baseline. Nothing on. Read val_l1/all_dims vs 0.044 (= copy current pose).
env $BASE EXPT_NAME=e0_baseline $PHD

# --- loss / target, one at a time ---
env $BASE EXPT_NAME=e1_weights   EXTRA_ARGS="$J --loss_dim_weight_mode file" $PHD
env $BASE EXPT_NAME=e2_deltas    EXTRA_ARGS="$J --predict_deltas" $PHD
env $BASE EXPT_NAME=e3_temporal  EXTRA_ARGS="$J --temporal_weight_mode horizon --action_horizon_for_weights 25" $PHD
env $BASE EXPT_NAME=e4_constants EXTRA_ARGS="$J --use_constant_dims" $PHD
env $BASE EXPT_NAME=e5_active45  EXTRA_ARGS="$J --action_dims_mode active --use_constant_dims" $PHD
#   ^ active NEEDS constant_dims to know which dims to drop -- the one unavoidable pairing

# --- compute, one at a time (short runs; read s/step, not val loss) ---
env $BASE EXPT_NAME=e6_tactile_input EXTRA_ARGS="$J --tactile_mode input" $PHD
env $BASE EXPT_NAME=e7_flash         EXTRA_ARGS="$J --explicit_flash_attn" $PHD
env $BASE EXPT_NAME=e8_crop192       EXTRA_ARGS="$J --image_crop 192" $PHD
env $BASE EXPT_NAME=e9_cam3          EXTRA_ARGS="$J --cameras head_left,wrist_right,wrist_left" $PHD
env $BASE EXPT_NAME=e10_batch96      EXTRA_ARGS="$J --batch_size 96" $PHD

# short form for the compute ones (minutes, not hours)
env $BASE EXPT_NAME=e6_speed \
  EXTRA_ARGS="$J --tactile_mode input --num_epochs 2 --max_train_steps 60 --max_val_steps 5" $PHD

# --- E11 combine ONLY what won, after reading E1-E10 ---
env $BASE EXPT_NAME=e11_combined \
  EXTRA_ARGS="$J --loss_dim_weight_mode file --predict_deltas --use_constant_dims \
              --temporal_weight_mode horizon --tactile_mode input --explicit_flash_attn" $PHD

# -----------------------------------------------------------------------------
# 3. READING RESULTS
# -----------------------------------------------------------------------------

# all_dims is the ONLY cross-variant number (radians, unweighted, all 65 cols).
# 0.044 = copy current pose,  0.103 = dataset mean.
grep -h "VAL epoch" $R/ckpt_dir/fold_plane/<run>/ori_debug_rank0.log | tail -10

# loss COMPONENTS -- a stalled total hides which term is stuck (l1 vs kl vs l1_tac)
grep -hoE "l1=[0-9.]+|kl=[0-9.]+|l1_tac=[0-9.]+" $R/ckpt_dir/fold_plane/<run>/ori_debug_rank0.log | tail -30

# step time
tail -5 $R/ckpt_dir/fold_plane/<run>/dataloader_timing.log

# -----------------------------------------------------------------------------
# 4. DEPLOY
# -----------------------------------------------------------------------------
cd $SUB && ./scripts/sync_vendor.sh && ./scripts/contract_local.sh

# inference-only overrides (need NOT match training)
#   VITAC_EXPLICIT_FLASH_ATTN=1   fused attention at deploy
#   VITAC_SMOOTHING=auto          chunk-seam smoothing (default)
#   VITAC_OPTIMIZATION=none       skip torch.compile

# -----------------------------------------------------------------------------
# 5. GIT / CRLF
# -----------------------------------------------------------------------------

# symptom:  /bin/bash^M: bad interpreter: No such file or directory
# cause:    core.autocrlf=input only normalises on COMMIT, not checkout.
# fix:      .gitattributes (committed at repo root) forces eol=lf for everyone.

cd $HOME/other/ori/ori && git add --renormalize . && git status   # review, then commit

file $R/*.sh | grep CRLF                 # silence = clean
sed -i 's/\r$//' $R/run_ori_job.sh       # fix one file, no extra package
bash <(tr -d '\r' < $R/run_ori_job.sh) 4 # run a CRLF script without editing it

# commit convention used so far
cd $HOME/other/ori/ori && git status --short && git log --oneline -5

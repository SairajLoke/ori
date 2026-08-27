#!/bin/bash
# Reference commands for phd run job submissions -- experiment matrix covering
# this session's new features (qpos masking, decision fusion, image history,
# torque input). Not meant to be run as one script (each block is a separate
# job submission) -- copy/paste the block you want.
#
# Add --save_attention_maps --attention_maps_n_samples 2 to any EXTRA_ARGS
# below if you also want temporal/self/between-modality attention maps saved
# during validation for that run (off by default, so none of these capture
# them as written).

NUM_GPUS=4

# 1. Baseline -- no new features, comparison point
EXPT_NAME=baseline_aug27 \
MIXED_PRECISION=bf16 MAX_EPISODES=0 USE_NORMALIZATION=0 \
phd run -ng $NUM_GPUS -p shr_gpu -GR H100 -l %J.log sh run_ori_job.sh $NUM_GPUS

# 2. qpos masking -- fixed mode, moderate probability
EXPT_NAME=qposmask_fixed_p05_aug27 \
MIXED_PRECISION=bf16 MAX_EPISODES=0 USE_NORMALIZATION=0 \
EXTRA_ARGS="--qpos_mask_prob 0.5 --qpos_mask_mode fixed" \
phd run -ng $NUM_GPUS -p shr_gpu -GR H100 -l %J.log sh run_ori_job.sh $NUM_GPUS

# 3. qpos masking -- full ablation (vision/tactile-only baseline, qpos never reaches the model)
EXPT_NAME=qposmask_fixed_p10_aug27 \
MIXED_PRECISION=bf16 MAX_EPISODES=0 USE_NORMALIZATION=0 \
EXTRA_ARGS="--qpos_mask_prob 1.0 --qpos_mask_mode fixed" \
phd run -ng $NUM_GPUS -p shr_gpu -GR H100 -l %J.log sh run_ori_job.sh $NUM_GPUS

# 4. qpos masking -- static_adaptive mode (masks harder when qpos is near-static)
EXPT_NAME=qposmask_adaptive_aug27 \
MIXED_PRECISION=bf16 MAX_EPISODES=0 USE_NORMALIZATION=0 \
EXTRA_ARGS="--qpos_mask_prob 0.5 --qpos_mask_mode static_adaptive --qpos_static_velocity_threshold 0.01" \
phd run -ng $NUM_GPUS -p shr_gpu -GR H100 -l %J.log sh run_ori_job.sh $NUM_GPUS

# 5. Decision fusion -- image-history + tactile, torque excluded (isolates torque's contribution later)
EXPT_NAME=decisionfusion_notorque_aug27 \
MIXED_PRECISION=bf16 MAX_EPISODES=0 USE_NORMALIZATION=0 \
EXTRA_ARGS="--use_decision_fusion --image_history" \
phd run -ng $NUM_GPUS -p shr_gpu -GR H100 -l %J.log sh run_ori_job.sh $NUM_GPUS

# 6. Decision fusion -- full (image-history + tactile + torque)
EXPT_NAME=decisionfusion_full_aug27 \
MIXED_PRECISION=bf16 MAX_EPISODES=0 USE_NORMALIZATION=0 \
EXTRA_ARGS="--use_decision_fusion --image_history --torque_input" \
phd run -ng $NUM_GPUS -p shr_gpu -GR H100 -l %J.log sh run_ori_job.sh $NUM_GPUS

# 7. Decision fusion + qpos masking combined
EXPT_NAME=decisionfusion_qposmask_aug27 \
MIXED_PRECISION=bf16 MAX_EPISODES=0 USE_NORMALIZATION=0 \
EXTRA_ARGS="--use_decision_fusion --image_history --torque_input --qpos_mask_prob 0.5 --qpos_mask_mode fixed" \
phd run -ng $NUM_GPUS -p shr_gpu -GR H100 -l %J.log sh run_ori_job.sh $NUM_GPUS

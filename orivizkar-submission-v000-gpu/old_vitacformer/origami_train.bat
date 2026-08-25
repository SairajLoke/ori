python origami_imitate_episodes.py ^
--task_name fold_plane ^
--expt_name jul12_2min_unmasked_unnormed_12epis_continued2 ^
--policy_class ACT --kl_weight 10 --chunk_size 00 --hidden_dim 512 ^
--batch_size 2 --dim_feedforward 3200 ^
--ckpt_save_epochs 1 ^
--num_epochs 1000 ^
--lr 3e-4 ^
--seed 0 ^
--use_tactile ^
--state_dim 65 ^
--visualize_batch ^
--doImageTransforms ^
--ckpt_dir ckpt_dir/fold_plane 

@REM --resume_path "/home/sr5/sairaj.loke/other/ori/ori/ViTacFormer/ckpt_dir/fold_plane/jul12_2min_unmasked_unnormed_12epis_20260712_232005_ori_tactile/policy_epoch_45_loss_0.046.ckpt"

REM #2000
@echo off
@REM REM Run from this scripts own directory so relative paths resolve correctly
@REM cd /d "%~dp0"

@REM cd dataset\ha_data 
@REM pip install -e . 
@REM cd ..\..

@REM pip install -e D:\ORI\ori\ViTacFormer\detr
python3 origami_imitate_episodes.py \
--task_name fold_plane \
--ckpt_dir ckpt_dir/fold_plane \
--policy_class ACT --kl_weight 10 --chunk_size 00 --hidden_dim 512 \
--batch_size 00 --dim_feedforward 3200 \
--num_epochs 2000  --lr 1e-4 \
--seed 0 \
--use_tactile \
# --resume_path ~
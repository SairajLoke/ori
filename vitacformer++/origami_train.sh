
# cd dataset/ha_data && pip install -e . && cd ../..
# cd detr && pip install -e . && cd ..

afafawwafw
python3 origami_imitate_episodes.py \
--task_name fold_plane \
--expt_name aug3_unmasked_unnormed_99epis_pretrainedjul12-u-u-12epi-e30 \
--policy_class ACT --kl_weight 10 --chunk_size 00 --hidden_dim 512 \
--batch_size 128 --dim_feedforward 3200 \
--ckpt_save_epochs 1 \
--num_epochs 1000  \
--lr 3e-4 \
--seed 0 \
--use_tactile \
--state_dim 65 \
--visualize_batch \
--doImageTransforms \
--ckpt_dir ckpt_dir/fold_plane \
--load_pretrained_for_newtraining "/home/sr5/sairaj.loke/other/ori/ori/ViTacFormer/ckpt_dir/fold_plane/jul12_unmasked_unnormed_12epis_continued1_20260713_222821_ori_tactile/policy_epoch_30_loss_0.041.ckpt"
# \
# --resume_path "/home/sr5/sairaj.loke/other/ori/ori/ViTacFormer/ckpt_dir/fold_plane/jul12_unmasked_unnormed_12epis_continued1_20260713_222821_ori_tactile/policy_epoch_30_loss_0.041.ckpt"

#2000
# jul12_2min_unmasked_unnormed_12epis_continued2
#batch size 128
# 'fp16'
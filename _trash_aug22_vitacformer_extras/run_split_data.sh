
# Split by specific episode indices
HF_HUB_OFFLINE=1 lerobot-edit-dataset \
    --root "/home/sr5/sairaj.loke/other/data" \
    --repo_id full_ori_12 \
    --operation.type split \
    --operation.splits '{"train": [0,1,2,3,4,5,6,7], "test": [8, 9], "val": [10,11] } ' \
    --push_to_hub false



#NOTE TESTED YET
HF_HUB_OFFLINE=1 lerobot-edit-dataset \
    --new_root "/home/sr5/sairaj.loke/other/data/full_ori_12" \
    --new_repo_id full_ori_12 \
    --operation.type merge \
    --operation.roots "['/home/sr5/sairaj.loke/other/data/season_POC22032_2026_05_14_19_21_01_train/lerobot3.0', 
                        '/home/sr5/sairaj.loke/other/data/season_POC22032_2026_05_14_20_40_58_train/lerobot3.0' ]" \
    --operation.repo_ids "['season_POC22032_2026_05_14_19_21_01_train/lerobot3.0', 
                           'season_POC22032_2026_05_14_20_40_58_train/lerobot3.0' ]" \
    --push_to_hub false

                        #    'season_POC22032_2026_05_14_21_08_06_train/lerobot3.0',
                        #    'season_POC22032_2026_05_15_16_43_23_train/lerobot3.0', ]"


                        
    # --root "/home/sr5/sairaj.loke/other/data" \

rootdir="/home/sai/Desktop/ORI/ori/stage_annotator/auto_progress"

.venv/bin/python fold_progress_pipeline.py \
    --fold_dir /home/sai/Desktop/ORI/ori/stage_annotator/auto_progress/fold0_frames \
    --fold_name "triangle_fold_1" \
    --fold_description "bringing left wing edge to align with center crease" \
    --anchor_start "$rootdir"/anchors/episode_*/b0.jpg \
    --anchor_end "$rootdir"/anchors/episode_*/b1.jpg


    #     --out fold1_dryrun.json \
    # --max_batches 1
#!/bin/bash
# Runs fold_progress_pipeline.py for folds 1-5 (fold 0 already scored -- skipped
# here on purpose so we never re-spend API quota re-doing it), pulling fold_name/
# fold_description from auto_progress/fold_meta.json instead of hardcoding them
# here -- fold_meta.json is the single source of truth (edit descriptions there,
# not in this script). Anchors are pooled across every episode exported so far
# (auto_progress/anchors/episode_*/bN.jpg), not just one episode.
#
# --sample_per_episode 20 (default is 10) -- denser sampling per your request,
# giving 6*20=120 frames/fold -> 10 batches/fold at BATCH_SIZE=12. Adjust down
# if your fresh API key's daily quota turns out tighter than that; the
# DailyQuotaExhausted fix will now fail fast (not burn retries) if you hit it.
set -e

rootdir="/home/sai/Desktop/ORI/ori/stage_annotator/auto_progress"
meta="$rootdir/fold_meta.json"

for fold in 1 2 3 4 5; do
    b_start=$fold
    b_end=$((fold + 1))

    mapfile -t fields < <(.venv/bin/python3 -c "
import json
d = json.load(open('$meta'))['$fold']
print(d['fold_name'])
print(d['fold_description'])
")
    fold_name="${fields[0]}"
    fold_description="${fields[1]}"

    echo "=== fold $fold: $fold_name (b$b_start -> b$b_end) ==="
    .venv/bin/python fold_progress_pipeline.py \
        --fold_dir "$rootdir/fold${fold}_frames" \
        --fold_name "$fold_name" \
        --fold_description "$fold_description" \
        --anchor_start "$rootdir"/anchors/episode_*/b${b_start}.jpg \
        --anchor_end "$rootdir"/anchors/episode_*/b${b_end}.jpg \
        --sample_per_episode 20 \
        --out "fold${fold}_progress_results.json"
    echo
done

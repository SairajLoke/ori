"""
Remove feature entries from a LeRobot v3.0 dataset's meta/info.json (and
meta/stats.json, if present).

Separate from reshape_videos.py on purpose: reshaping never removes anything,
it only resizes the video files you point it at. This script is for a
different, explicit decision -- declaring that a dataset copy no longer
carries a feature at all, e.g. after resizing only some camera keys and
deliberately not shipping video files for the rest.

Metadata-only. Never touches data/ or videos/ contents; if a video feature is
removed but its video files are still present on disk, they are simply
orphaned (harmless, just unused disk space) -- this script will not delete
them.

Usage:
    python remove_dataset_features.py \
        --dataset_root /path/to/lerobot3.0_xxx \
        --features observation.images.tactile_raw,observation.images.tactile_deform

Defaults to a dry run (prints what would change, writes nothing). Pass --apply
to actually write. A .bak copy of each file is kept next to the original
before it is overwritten.
"""

import argparse
import json
import shutil
from pathlib import Path


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_root", type=Path, required=True,
                    help="Root of the lerobot3.0 dataset (contains meta/info.json).")
    p.add_argument("--features", type=str, required=True,
                    help="Comma-separated feature keys to remove, e.g. "
                         "observation.images.tactile_raw,observation.images.tactile_deform")
    p.add_argument("--apply", action="store_true",
                    help="Actually write the changes. Without this flag, only prints "
                         "what would be removed.")
    args = p.parse_args()

    dataset_root = args.dataset_root.resolve()
    features = [f.strip() for f in args.features.split(",") if f.strip()]
    if not features:
        raise SystemExit("--features is empty.")

    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise SystemExit(f"{info_path} not found.")
    stats_path = dataset_root / "meta" / "stats.json"

    info = load_json(info_path)
    stats = load_json(stats_path) if stats_path.exists() else None

    in_info = [f for f in features if f in info.get("features", {})]
    not_in_info = [f for f in features if f not in info.get("features", {})]
    in_stats = [f for f in features if stats is not None and f in stats]

    print(f"dataset_root : {dataset_root}")
    print(f"features     : {features}")
    print(f"will remove from info.json  : {in_info}")
    if not_in_info:
        print(f"not found in info.json      : {not_in_info}  (no-op for these)")
    if stats is not None:
        print(f"will remove from stats.json : {in_stats}")
    else:
        print("stats.json not found, nothing to remove there")

    is_video = {f: info["features"][f].get("dtype") == "video" for f in in_info}
    for f in in_info:
        if is_video[f]:
            video_dir = dataset_root / "videos" / f
            if video_dir.exists():
                print(f"[NOTE] {f} is a video feature and {video_dir} still exists on disk. "
                      f"This script only edits metadata -- those files are left in place, "
                      f"just no longer referenced by meta/info.json.")

    if not args.apply:
        print("\nDry run only (no files written). Re-run with --apply to write.")
        return

    if in_info:
        shutil.copy2(info_path, info_path.with_suffix(".json.bak"))
        for f in in_info:
            del info["features"][f]
        save_json(info_path, info)
        print(f"\n[WRITE] {info_path}  (backup: {info_path.with_suffix('.json.bak')})")

    if stats is not None and in_stats:
        shutil.copy2(stats_path, stats_path.with_suffix(".json.bak"))
        for f in in_stats:
            del stats[f]
        save_json(stats_path, stats)
        print(f"[WRITE] {stats_path}  (backup: {stats_path.with_suffix('.json.bak')})")

    print("\nDone.")


if __name__ == "__main__":
    main()

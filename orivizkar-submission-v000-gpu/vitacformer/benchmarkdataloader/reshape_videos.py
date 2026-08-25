"""
Resize all videos in a LeRobot v3.0 dataset to a target resolution.

This script resizes video frames to the resolution expected by the model
(320x224 as per origami_dataset.py), so runtime resizing during training
becomes unnecessary.

Usage:
    python reshape_videos.py \
        --dataset_root /path/to/lerobot3.0 \
        --output_root  /path/to/lerobot3.0_resized \
        --resize_width 320 \
        --resize_height 224 \
        --workers 4

Always inspect --dry-run output first on a new dataset layout before running
for real.
"""

# python3 reshape_videos.py \
#   --dataset_root /media/sai/CRUZER_BLA/ori/dataset/season_POC22061_2026_07_09_16_23_46_train/lerobot3.0_shortgop15 \
#   --output_root  /media/sai/CRUZER_BLA/ori/dataset/season_POC22061_2026_07_09_16_23_46_train/lerobot3.0_shortgop15_224 \
#   --resize_width 224 --resize_height 224 \
#   --video_keys observation.images.head_left,observation.images.head_right,observation.images.wrist_left,observation.images.wrist_right \
#   --workers 3


import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "logs"


class _Tee:
    """Mirrors writes to the original stream and a log file."""

    def __init__(self, stream, log_path: Path):
        self._stream = stream
        self._file = open(log_path, "a")

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()


@dataclass
class VideoJob:
    video_key: str
    rel_path: Path          # relative to dataset_root
    src: Path
    dst: Path
    src_width: int
    src_height: int


def run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def ffprobe_frame_count(path: Path) -> int:
    """Exact decoded frame count."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0", str(path),
    ]
    result = run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe frame count failed for {path}: {result.stderr.strip()}")
    out = result.stdout.strip()
    if not out or not out.isdigit():
        raise RuntimeError(f"ffprobe returned unparseable frame count for {path}: {out!r}")
    return int(out)


def ffprobe_video_props(path: Path) -> dict:
    """Get width, height, codec, etc."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,codec_name,pix_fmt",
        "-of", "json", str(path),
    ]
    result = run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe stream info failed for {path}: {result.stderr.strip()}")
    return json.loads(result.stdout)["streams"][0]


def discover_video_jobs(dataset_root: Path, output_root: Path,
                        video_keys: list[str] | None = None) -> list[VideoJob]:
    """
    Walk <dataset_root>/videos/<video_key>/chunk-*/file-*.mp4 and create jobs.

    video_keys: if given, only these video-key subdirectories are processed
    (others under videos/ are left out of output_root entirely). Use this to
    skip streams the model never decodes, e.g. tactile_raw/tactile_deform,
    which origami_dataset.py prunes from ds.meta.features before training.
    """
    videos_dir = dataset_root / "videos"
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"Expected a 'videos' directory at {videos_dir}, not found.")

    jobs: list[VideoJob] = []
    video_key_dirs = sorted(p for p in videos_dir.iterdir() if p.is_dir())
    if not video_key_dirs:
        raise FileNotFoundError(f"No video-key subdirectories found under {videos_dir}")

    if video_keys:
        wanted = set(video_keys)
        found = {p.name for p in video_key_dirs}
        missing = wanted - found
        if missing:
            raise FileNotFoundError(
                f"--video_keys requested {sorted(missing)} but only found: {sorted(found)}"
            )
        video_key_dirs = [p for p in video_key_dirs if p.name in wanted]

    for key_dir in video_key_dirs:
        video_key = key_dir.name
        mp4_files = sorted(key_dir.glob("chunk-*/file-*.mp4"))
        if not mp4_files:
            print(f"[WARN] No chunk-*/file-*.mp4 files found under {key_dir} — skipping this key.")
            continue
        for src in mp4_files:
            props = ffprobe_video_props(src)
            rel_path = src.relative_to(dataset_root)
            dst = output_root / rel_path
            jobs.append(VideoJob(
                video_key=video_key,
                rel_path=rel_path,
                src=src,
                dst=dst,
                src_width=props.get("width", 0),
                src_height=props.get("height", 0),
            ))
    return jobs


def copy_non_video_tree(dataset_root: Path, output_root: Path):
    """Copy data/ and meta/ (meta will be updated after resizing)."""
    # Copy data/ unchanged
    src = dataset_root / "data"
    dst = output_root / "data"
    if not src.is_dir():
        raise FileNotFoundError(f"Expected 'data' directory at {src}, not found.")
    if dst.exists():
        print(f"[SKIP] {dst} already exists, leaving as-is.")
    else:
        print(f"[COPY] {src} -> {dst}")
        shutil.copytree(src, dst)
    
    # Copy meta/ (will be updated after video resizing)
    src = dataset_root / "meta"
    dst = output_root / "meta"
    if not src.is_dir():
        raise FileNotFoundError(f"Expected 'meta' directory at {src}, not found.")
    if dst.exists():
        print(f"[SKIP] {dst} already exists, will update info.json in place.")
    else:
        print(f"[COPY] {src} -> {dst}")
        shutil.copytree(src, dst)


def update_metadata(output_root: Path, resize_width: int, resize_height: int, video_keys: list[str]):
    """
    Update meta/info.json with new video dimensions after resizing.

    LeRobot v3 stores video shape as [height, width, channels] and also
    has video.height and video.width in the info dict.
    """
    info_path = output_root / "meta" / "info.json"
    if not info_path.exists():
        print(f"[WARN] {info_path} not found, skipping metadata update.")
        return
    
    with open(info_path, "r") as f:
        info = json.load(f)
    
    updated = False
    if "features" in info:
        for key in video_keys:
            if key in info["features"]:
                feat = info["features"][key]
                if isinstance(feat, dict) and "shape" in feat:
                    old_shape = feat["shape"]
                    # Shape is [height, width, channels] for LeRobot v3
                    if len(old_shape) == 3:
                        new_shape = [resize_height, resize_width, old_shape[2]]
                        feat["shape"] = new_shape
                        print(f"[META] Updated {key} shape: {old_shape} -> {new_shape}")
                        
                        # Also update info dict if present
                        if "info" in feat:
                            if "video.height" in feat["info"]:
                                feat["info"]["video.height"] = resize_height
                            if "video.width" in feat["info"]:
                                feat["info"]["video.width"] = resize_width
                            print(f"[META] Updated {key} info: height={resize_height}, width={resize_width}")
                        updated = True
    
    if updated:
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"[META] Saved updated {info_path}")
    else:
        print(f"[META] No features updated (check if video keys match)")


def process_job(job: VideoJob, resize_width: int, resize_height: int,
                crf: int, preset: str, ffmpeg_threads: int,
                dry_run: bool) -> dict:
    """Resize a video to the target resolution."""
    job.dst.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        return {"job": job, "status": "would_resize",
                "src_size": f"{job.src_width}x{job.src_height}",
                "dst_size": f"{resize_width}x{resize_height}"}

    src_frames = ffprobe_frame_count(job.src)

    # Build ffmpeg command with resize filter
    cmd = [
        "ffmpeg", "-y", "-i", str(job.src),
        "-map", "0:v:0",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-vf", f"scale={resize_width}:{resize_height}",
        "-pix_fmt", "yuv420p",
        "-vsync", "cfr",
    ]
    if ffmpeg_threads:
        cmd += ["-threads", str(ffmpeg_threads)]
    cmd += [str(job.dst)]

    t0 = time.time()
    result = run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        job.dst.unlink(missing_ok=True)
        return {"job": job, "status": "ffmpeg_failed", "stderr": result.stderr[-4000:]}

    dst_frames = ffprobe_frame_count(job.dst)
    if dst_frames != src_frames:
        job.dst.unlink(missing_ok=True)
        return {
            "job": job, "status": "frame_count_mismatch",
            "src_frames": src_frames, "dst_frames": dst_frames,
        }

    # Get output dimensions
    dst_props = ffprobe_video_props(job.dst)
    
    return {
        "job": job, "status": "ok", "elapsed": elapsed,
        "src_frames": src_frames, "dst_frames": dst_frames,
        "src_size": job.src.stat().st_size, "dst_size": job.dst.stat().st_size,
        "src_resolution": f"{job.src_width}x{job.src_height}",
        "dst_resolution": f"{dst_props.get('width', 0)}x{dst_props.get('height', 0)}",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_root", type=Path, required=True,
                    help="Root of the source lerobot3.0 dataset (contains data/, meta/, videos/).")
    p.add_argument("--output_root", type=Path, required=True,
                    help="Destination root for the resized dataset (created fresh).")
    p.add_argument("--resize_width", type=int, default=320,
                    help="Target video width (default: 320, as required by model).")
    p.add_argument("--resize_height", type=int, default=224,
                    help="Target video height (default: 224, as required by model).")
    p.add_argument("--video_keys", type=str, default=None,
                    help="Comma-separated video keys to resize (e.g. "
                         "observation.images.head_left,observation.images.head_right). "
                         "Default: all keys found under videos/. Restrict to the RGB "
                         "cameras to skip tactile_raw/tactile_deform, which "
                         "origami_dataset.py never decodes.")
    p.add_argument("--crf", type=int, default=20,
                    help="x264 CRF (quality). Lower = higher quality/larger file. Default: 20.")
    p.add_argument("--preset", type=str, default="veryfast",
                    help="x264 preset (speed/compression trade-off). Default: veryfast.")
    p.add_argument("--workers", type=int, default=4,
                    help="Number of videos to resize in parallel. Default: 4.")
    p.add_argument("--ffmpeg_threads", type=int, default=0,
                    help="Threads per ffmpeg process (0 = ffmpeg auto-detect).")
    p.add_argument("--dry_run", action="store_true",
                    help="Print the discovered job plan without running ffmpeg.")
    p.add_argument("--log_file", type=Path, default=None,
                    help="Path to also write all output to (mirrors stdout/stderr).")
    p.add_argument("--no_log", action="store_true",
                    help="Disable writing a log file (stdout only).")
    args = p.parse_args()

    if not args.no_log:
        log_file = args.log_file
        if log_file is None:
            DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
            log_file = DEFAULT_LOG_DIR / f"reshape_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        else:
            log_file.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = _Tee(sys.stdout, log_file)
        sys.stderr = _Tee(sys.stderr, log_file)
        print(f"[INFO] Logging to: {log_file}")

    if args.video_keys:
        args.video_keys = [k.strip() for k in args.video_keys.split(",") if k.strip()]

    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()

    if dataset_root == output_root:
        print("[ERROR] --output_root must differ from --dataset_root.", file=sys.stderr)
        sys.exit(1)
    if not dataset_root.is_dir():
        print(f"[ERROR] dataset_root does not exist: {dataset_root}", file=sys.stderr)
        sys.exit(1)
    for sub in ["data", "meta", "videos"]:
        if not (dataset_root / sub).is_dir():
            print(f"[ERROR] dataset_root is missing expected subdir '{sub}': {dataset_root / sub}",
                  file=sys.stderr)
            sys.exit(1)

    print(f"[INFO] dataset_root : {dataset_root}")
    print(f"[INFO] output_root  : {output_root}")
    print(f"[INFO] resize to={args.resize_width}x{args.resize_height} "
          f"crf={args.crf} preset={args.preset} workers={args.workers}")

    jobs = discover_video_jobs(dataset_root, output_root, video_keys=args.video_keys)
    by_key: dict[str, int] = {}
    for j in jobs:
        by_key[j.video_key] = by_key.get(j.video_key, 0) + 1
    print("[INFO] Discovered video files per key:")
    for k, n in sorted(by_key.items()):
        print(f"         {k:45s} {n:4d} files  [RESIZE]")
    print(f"[INFO] Total video files: {len(jobs)}")

    if args.dry_run:
        print("\n[DRY RUN] No files will be written. Job list:")
        for j in jobs:
            print(f"  RESIZE {j.rel_path}  {j.src_width}x{j.src_height} -> {args.resize_width}x{args.resize_height}")
        print("\n[DRY RUN] data/ and meta/ would be copied unchanged.")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    copy_non_video_tree(dataset_root, output_root)

    results = []
    failures = []
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                process_job, j, args.resize_width, args.resize_height,
                args.crf, args.preset, args.ffmpeg_threads, args.dry_run,
            ): j
            for j in jobs
        }
        done = 0
        for fut in as_completed(futures):
            j = futures[fut]
            done += 1
            try:
                r = fut.result()
            except NotImplementedError as e:
                print(f"[ERROR] {e}")
                ex.shutdown(cancel_futures=True)
                sys.exit(1)
            results.append(r)
            status = r["status"]
            if status == "ok":
                shrink = 100.0 * (1 - r["dst_size"] / r["src_size"])
                print(f"[{done}/{len(jobs)}] OK  {j.rel_path}  "
                      f"{r['elapsed']:.1f}s  frames={r['dst_frames']} (match)  "
                      f"size {r['src_size']/1e6:.0f}MB -> {r['dst_size']/1e6:.0f}MB ({shrink:+.0f}%)  "
                      f"res {r['src_resolution']} -> {r['dst_resolution']}")
            else:
                failures.append(r)
                print(f"[{done}/{len(jobs)}] FAIL {j.rel_path}  status={status}  {r}")

    elapsed_total = time.time() - t_start
    n_ok = sum(1 for r in results if r["status"] == "ok")
    print("\n" + "=" * 70)
    print(f"Done in {elapsed_total:.1f}s — {n_ok} resized, {len(failures)} failed.")
    if failures:
        print("FAILURES (output_root is INCOMPLETE — do not point training at it):")
        for r in failures:
            print(f"  {r['job'].rel_path}: {r['status']}")
        sys.exit(1)

    # Update metadata with new dimensions
    video_keys = list(by_key.keys())
    update_metadata(output_root, args.resize_width, args.resize_height, video_keys)

    print("\nNext: point DATASET_ROOT / FULL_DATASET at the new output_root and re-run training.")


if __name__ == "__main__":
    main()


# python reshape_videos.py \
#   --dataset_root /home/sr5/sairaj.loke/other/new_data/larger_data_shortgop \
#   --output_root /home/sr5/sairaj.loke/other/new_data/larger_data_shortgop_resized \
#   --resize_width 320 \
#   --resize_height 224 \
#   --workers 4
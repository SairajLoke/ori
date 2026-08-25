"""
Re-encode a LeRobot v3.0 dataset's videos with a short GOP (keyframe interval).

Why: torchcodec's random-access seeking (used by LeRobotDataset under
shuffle=True) has to decode forward from the nearest preceding keyframe to
reach a requested frame. Source videos recorded with ffmpeg defaults have a
GOP of ~240+ frames (~8s @ 30fps), so a single random frame read can require
decoding hundreds of throwaway frames, 4x per sample (one per camera). This
script re-encodes every camera video with a short GOP so random access is
O(gop) instead of O(240+), leaving everything else (fps, frame count,
resolution, pixel format, file layout) untouched so no metadata needs to
change — LeRobot resolves frames by timestamp, not byte offset.

Only re-encodes the video keys actually consumed by training
(dataset/origami_dataset.py drops observation.images.tactile_raw and
observation.images.tactile_deform from ds.meta.features before any frame is
read). Those two keys are just copied through unchanged by default — pass
--include-all to re-encode them too.

Usage:
    python reencode_videos_shortgop.py \
        --dataset_root /path/to/lerobot3.0 \
        --output_root  /path/to/lerobot3.0_shortgop \
        --gop 15 --workers 4

Always inspect --dry-run output first on a new dataset layout before running
for real.
"""

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
    """Mirrors writes to the original stream and a log file. All of this
    script's output goes through print() in the main process (worker
    processes only return dicts, never print), so redirecting sys.stdout
    here captures everything without touching call sites."""

    def __init__(self, stream, log_path: Path):
        self._stream = stream
        self._file = open(log_path, "a")

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

DEFAULT_SKIP_KEYS = [
    "observation.images.tactile_raw",
    "observation.images.tactile_deform",
]


@dataclass
class VideoJob:
    video_key: str
    rel_path: Path          # relative to dataset_root, e.g. videos/<key>/chunk-000/file-000.mp4
    src: Path
    dst: Path
    reencode: bool           # False => plain copy (skip-listed key)


def run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def ffprobe_frame_count(path: Path) -> int:
    """Exact decoded frame count (not the possibly-stale container 'nb_frames' tag)."""
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


def ffprobe_keyframe_count(path: Path) -> tuple[int, int]:
    """Returns (total_frames, keyframe_count) via the key_frame flag stream."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=key_frame", "-of", "csv=p=0", str(path),
    ]
    result = run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe keyframe scan failed for {path}: {result.stderr.strip()}")
    flags = [line for line in result.stdout.strip().splitlines() if line != ""]
    total = len(flags)
    keyframes = sum(1 for f in flags if f.strip() == "1")
    return total, keyframes


def ffprobe_video_props(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,codec_name,pix_fmt",
        "-of", "json", str(path),
    ]
    result = run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe stream info failed for {path}: {result.stderr.strip()}")
    return json.loads(result.stdout)["streams"][0]


def discover_video_jobs(dataset_root: Path, output_root: Path, skip_keys: set[str]) -> list[VideoJob]:
    """
    Walk <dataset_root>/videos/<video_key>/chunk-*/file-*.mp4 without assuming
    which camera keys, how many chunks, or how many files per chunk exist.
    """
    videos_dir = dataset_root / "videos"
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"Expected a 'videos' directory at {videos_dir}, not found.")

    jobs: list[VideoJob] = []
    video_key_dirs = sorted(p for p in videos_dir.iterdir() if p.is_dir())
    if not video_key_dirs:
        raise FileNotFoundError(f"No video-key subdirectories found under {videos_dir}")

    for key_dir in video_key_dirs:
        video_key = key_dir.name
        mp4_files = sorted(key_dir.glob("chunk-*/file-*.mp4"))
        if not mp4_files:
            # Be loud rather than silently skipping an unexpected layout.
            print(f"[WARN] No chunk-*/file-*.mp4 files found under {key_dir} — skipping this key.")
            continue
        for src in mp4_files:
            rel_path = src.relative_to(dataset_root)
            dst = output_root / rel_path
            jobs.append(VideoJob(
                video_key=video_key,
                rel_path=rel_path,
                src=src,
                dst=dst,
                reencode=video_key not in skip_keys,
            ))
    return jobs


def copy_non_video_tree(dataset_root: Path, output_root: Path):
    """Copy data/ and meta/ unchanged — frame resolution is by timestamp, not
    byte offset, so re-encoding videos alone doesn't require touching these."""
    for sub in ["data", "meta"]:
        src = dataset_root / sub
        dst = output_root / sub
        if not src.is_dir():
            raise FileNotFoundError(f"Expected '{sub}' directory at {src}, not found.")
        if dst.exists():
            print(f"[SKIP] {dst} already exists, leaving as-is.")
            continue
        print(f"[COPY] {src} -> {dst}")
        shutil.copytree(src, dst)


def process_job(job: VideoJob, gop: int, crf: int, preset: str, ffmpeg_threads: int,
                 resize: bool, resize_width: int | None, resize_height: int | None,
                 dry_run: bool) -> dict:
    job.dst.parent.mkdir(parents=True, exist_ok=True)

    if resize:
        # Reserved for a future iteration — do not silently ignore the flag.
        raise NotImplementedError(
            "--resize was requested but resizing during re-encode is not implemented yet. "
            "This flag is a placeholder for a future iteration (would also require updating "
            "meta/info.json video.height/video.width/shape for the affected features). "
            "Re-run without --resize."
        )

    if not job.reencode:
        if dry_run:
            return {"job": job, "status": "would_copy"}
        shutil.copy2(job.src, job.dst)
        return {"job": job, "status": "copied"}

    if dry_run:
        return {"job": job, "status": "would_reencode"}

    src_frames = ffprobe_frame_count(job.src)

    cmd = [
        "ffmpeg", "-y", "-i", str(job.src),
        "-map", "0:v:0",  # video stream only — drop audio/other streams if present
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-g", str(gop),
        "-keyint_min", str(gop),
        "-sc_threshold", "0",   # disable scene-cut adaptive keyframes so GOP stays fixed
        "-pix_fmt", "yuv420p",
        "-vsync", "cfr",        # force constant frame rate; broadly supported (incl. old ffmpeg)
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
        # Frame-count drift silently desyncs frame i from its timestamp/action label.
        job.dst.unlink(missing_ok=True)
        return {
            "job": job, "status": "frame_count_mismatch",
            "src_frames": src_frames, "dst_frames": dst_frames,
        }

    return {
        "job": job, "status": "ok", "elapsed": elapsed,
        "src_frames": src_frames, "dst_frames": dst_frames,
        "src_size": job.src.stat().st_size, "dst_size": job.dst.stat().st_size,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_root", type=Path, required=True,
                    help="Root of the source lerobot3.0 dataset (contains data/, meta/, videos/).")
    p.add_argument("--output_root", type=Path, required=True,
                    help="Destination root for the re-encoded dataset (created fresh).")
    p.add_argument("--gop", type=int, default=15,
                    help="Keyframe interval in frames for re-encoded videos (default: 15, ~0.5s @30fps).")
    p.add_argument("--crf", type=int, default=20,
                    help="x264 CRF (quality). Lower = higher quality/larger file. Default: 20.")
    p.add_argument("--preset", type=str, default="veryfast",
                    help="x264 preset (speed/compression trade-off). Default: veryfast.")
    p.add_argument("--workers", type=int, default=4,
                    help="Number of videos to re-encode in parallel. Default: 4.")
    p.add_argument("--ffmpeg_threads", type=int, default=0,
                    help="Threads per ffmpeg process (0 = ffmpeg auto-detect). "
                         "Set explicitly to avoid oversubscribing CPUs when --workers > 1.")
    p.add_argument("--skip_video_keys", type=str, default=",".join(DEFAULT_SKIP_KEYS),
                    help="Comma-separated video keys to copy through unchanged instead of "
                         f"re-encoding (default: unused-by-training keys: {DEFAULT_SKIP_KEYS}).")
    p.add_argument("--include_all", action="store_true",
                    help="Re-encode every video key, ignoring --skip_video_keys.")
    p.add_argument("--resize", action="store_true",
                    help="(NOT IMPLEMENTED YET) Reserved flag for a future iteration that "
                         "resizes frames during re-encode. Passing this currently raises an error.")
    p.add_argument("--resize_width", type=int, default=None, help="Reserved for future --resize support.")
    p.add_argument("--resize_height", type=int, default=None, help="Reserved for future --resize support.")
    p.add_argument("--dry_run", action="store_true",
                    help="Print the discovered job plan without running ffmpeg or copying anything.")
    p.add_argument("--log_file", type=Path, default=None,
                    help="Path to also write all output to (mirrors stdout/stderr). "
                         f"Default: {DEFAULT_LOG_DIR}/reencode_<timestamp>.log. Pass --no_log to disable.")
    p.add_argument("--no_log", action="store_true",
                    help="Disable writing a log file (stdout only).")
    args = p.parse_args()

    if not args.no_log:
        log_file = args.log_file
        if log_file is None:
            DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
            log_file = DEFAULT_LOG_DIR / f"reencode_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        else:
            log_file.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = _Tee(sys.stdout, log_file)
        sys.stderr = _Tee(sys.stderr, log_file)
        print(f"[INFO] Logging to: {log_file}")

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

    skip_keys = set() if args.include_all else {k.strip() for k in args.skip_video_keys.split(",") if k.strip()}

    print(f"[INFO] dataset_root : {dataset_root}")
    print(f"[INFO] output_root  : {output_root}")
    print(f"[INFO] gop={args.gop} crf={args.crf} preset={args.preset} workers={args.workers}")
    print(f"[INFO] skip_video_keys (copied, not re-encoded): {sorted(skip_keys) or '(none)'}")

    jobs = discover_video_jobs(dataset_root, output_root, skip_keys)
    by_key: dict[str, int] = {}
    for j in jobs:
        by_key[j.video_key] = by_key.get(j.video_key, 0) + 1
    print("[INFO] Discovered video files per key:")
    for k, n in sorted(by_key.items()):
        mode = "COPY" if k in skip_keys else "REENCODE"
        print(f"         {k:45s} {n:4d} files  [{mode}]")
    print(f"[INFO] Total video files: {len(jobs)}")

    if args.dry_run:
        print("\n[DRY RUN] No files will be written. Job list:")
        for j in jobs:
            print(f"  {'COPY' if not j.reencode else 'REENCODE':9s} {j.rel_path}")
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
                process_job, j, args.gop, args.crf, args.preset, args.ffmpeg_threads,
                args.resize, args.resize_width, args.resize_height, args.dry_run,
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
                      f"size {r['src_size']/1e6:.0f}MB -> {r['dst_size']/1e6:.0f}MB ({shrink:+.0f}%)")
            elif status == "copied":
                print(f"[{done}/{len(jobs)}] COPY {j.rel_path}")
            else:
                failures.append(r)
                print(f"[{done}/{len(jobs)}] FAIL {j.rel_path}  status={status}  {r}")

    elapsed_total = time.time() - t_start
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_copied = sum(1 for r in results if r["status"] == "copied")
    print("\n" + "=" * 70)
    print(f"Done in {elapsed_total:.1f}s — {n_ok} re-encoded, {n_copied} copied, {len(failures)} failed.")
    if failures:
        print("FAILURES (output_root is INCOMPLETE — do not point training at it):")
        for r in failures:
            print(f"  {r['job'].rel_path}: {r['status']}")
        sys.exit(1)

    # Spot-check GOP reduction on a sample of re-encoded files.
    sample = [r for r in results if r["status"] == "ok"][:3]
    if sample:
        print("\nGOP verification (sample of re-encoded files):")
        for r in sample:
            total, kf = ffprobe_keyframe_count(r["job"].dst)
            avg_gop = total / kf if kf else float("inf")
            print(f"  {r['job'].rel_path}: {total} frames, {kf} keyframes, avg GOP = {avg_gop:.1f}")

    print("\nNext: point DATASET_ROOT / FULL_DATASET at the new output_root and re-run training, "
          "checking dataloader_timing.log for the 'dataloader' component before/after.")


if __name__ == "__main__":
    main()

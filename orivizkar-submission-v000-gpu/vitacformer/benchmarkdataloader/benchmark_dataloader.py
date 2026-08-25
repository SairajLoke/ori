"""
Load a LeRobot v3.0 dataset root exactly the way origami_imitate_episodes.py /
origami_dataset.get_origami_full_dataset does, iterate every batch for one
full epoch, and log per-batch + summary timings.

Purpose: compare raw dataloader speed between an original dataset root and a
short-GOP re-encoded copy (see reencode_videos_shortgop.py), with everything
else (delta_timestamps, tolerance, batch size, num_workers, shuffle) held
identical, so the only variable is the video encoding.

This intentionally does NOT do convert_batch / normalization / GPU transfer —
it isolates raw DataLoader.__next__() time (parquet read + video decode +
collation + pin_memory), which is the thing the re-encode targets.

Usage:
    python benchmark_dataloader.py \
        --dataset_root /path/to/lerobot3.0 \
        --label og_root \
        --log_dir /path/to/bench_logs \
        --video_backend pyav \
        --batch_size 16 --num_workers 4
"""

import argparse
import statistics
import sys
import time
import traceback
from pathlib import Path

# ViTacFormer root (this file lives in <root>/dataset/) — needed to import configs.py
_VITAC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_VITAC_ROOT))

DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "logs"

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def build_dataset(dataset_root: Path, video_backend: str, use_tactile: bool):
    """Mirrors dataset/origami_dataset.py:get_origami_full_dataset, parametrized
    on video_backend (that function hardcodes 'torchcodec', which fails to load
    on this machine — see benchmark run notes)."""
    import configs  # noqa: E402  (module-level prints are expected/harmless)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    meta_path = dataset_root / "meta" / "info.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found for dataset_root={dataset_root}")

    _episodes = list(range(configs.MAX_EPISODES)) if configs.MAX_EPISODES > 0 else None

    ds = LeRobotDataset(
        repo_id=None,
        root=dataset_root,
        image_transforms=None,
        delta_timestamps=configs.DELTA_TIMESTAMPS,
        video_backend=video_backend,
        tolerance_s=configs.TOLERANCE,
        episodes=_episodes,
    )

    # Same pruning origami_dataset.py does: these two video keys are never
    # consumed downstream, so don't pay decode cost for them.
    ignore_keys = ["observation.images.tactile_raw", "observation.images.tactile_deform"]
    for key in ignore_keys:
        if key in ds.meta.features:
            del ds.meta.features[key]

    if not use_tactile and "observation.tactile" in configs.DELTA_TIMESTAMPS:
        pass  # tactile stays in delta_timestamps regardless; harmless if unused downstream here

    return ds, configs


def percentile(sorted_vals, pct):
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * pct))
    return sorted_vals[idx]


def summarize(vals):
    if not vals:
        return {}
    s = sorted(vals)
    return {
        "n": len(vals),
        "avg": sum(vals) / len(vals),
        "p50": percentile(s, 0.50),
        "p95": percentile(s, 0.95),
        "min": s[0],
        "max": s[-1],
        "total": sum(vals),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_root", type=Path, required=True)
    p.add_argument("--label", type=str, required=True, help="Short tag identifying this run in logs/summary.")
    p.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR,
                    help=f"Directory to write {{label}}_batches.log / {{label}}_summary.log to. "
                         f"Default: {DEFAULT_LOG_DIR}")
    p.add_argument("--video_backend", type=str, default="pyav", choices=["pyav", "torchcodec"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--use_tactile", action="store_true", default=True)
    p.add_argument("--max_batches", type=int, default=None,
                    help="Cap number of batches (default: None = full epoch, every batch).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    batch_log_path = args.log_dir / f"{args.label}_batches.log"
    summary_log_path = args.log_dir / f"{args.label}_summary.log"

    def log_line(path, line):
        with open(path, "a") as f:
            f.write(line + "\n")

    with open(batch_log_path, "w") as f:
        f.write(f"# per-batch dataloader timing — label={args.label}\n")
        f.write(f"# dataset_root={args.dataset_root}\n")
        f.write(f"# video_backend={args.video_backend} batch_size={args.batch_size} "
                f"num_workers={args.num_workers} prefetch_factor={args.prefetch_factor}\n")

    t_setup_start = time.time()
    try:
        ds, configs_mod = build_dataset(args.dataset_root, args.video_backend, args.use_tactile)
    except Exception as e:
        log_line(summary_log_path, f"[FATAL] Dataset construction failed for {args.dataset_root}: {e}")
        log_line(summary_log_path, traceback.format_exc())
        print(f"[FATAL] {e}")
        sys.exit(1)
    t_setup = time.time() - t_setup_start

    dataloader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False,          # no GPU transfer in this benchmark — isolate decode/collate cost
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        drop_last=True,
    )

    total_batches_available = len(dataloader)
    n_target = min(args.max_batches, total_batches_available) if args.max_batches else total_batches_available

    header = (
        f"label={args.label} dataset_root={args.dataset_root} "
        f"video_backend={args.video_backend} total_frames={len(ds)} "
        f"batch_size={args.batch_size} num_workers={args.num_workers} "
        f"batches_available={total_batches_available} batches_target={n_target} "
        f"setup_time_s={t_setup:.2f}"
    )
    print(f"[{args.label}] {header}")
    log_line(summary_log_path, header)

    batch_times = []
    n_ok = 0
    error = None
    t_epoch_start = time.time()

    try:
        it = iter(dataloader)
        for batch_idx in range(n_target):
            t0 = time.time()
            batch = next(it)
            dt = time.time() - t0
            batch_times.append(dt)
            n_ok += 1

            log_line(batch_log_path, f"[BATCH {batch_idx}] dt={dt:.4f}s")

            if (batch_idx + 1) % 50 == 0:
                recent = batch_times[-50:]
                s = summarize(recent)
                summary_str = (
                    f"[{args.label}] progress {batch_idx + 1}/{n_target} "
                    f"last50: avg={s['avg']:.3f}s p50={s['p50']:.3f}s p95={s['p95']:.3f}s "
                    f"min={s['min']:.3f}s max={s['max']:.3f}s"
                )
                print(summary_str)
                log_line(batch_log_path, summary_str)

            # Sanity: confirm expected keys are present and non-empty (checks the
            # batch actually "loaded", not just that next() returned something).
            for required_key in ["observation.state", "action", "observation.images.head_left"]:
                if required_key not in batch:
                    raise KeyError(f"batch {batch_idx} missing expected key: {required_key}")
                if not hasattr(batch[required_key], "shape") or batch[required_key].shape[0] != args.batch_size:
                    raise ValueError(f"batch {batch_idx} key {required_key} has unexpected shape "
                                      f"{getattr(batch[required_key], 'shape', None)}")
    except StopIteration:
        pass
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        log_line(batch_log_path, f"[ERROR] Failed at batch {n_ok}: {error}")
        log_line(batch_log_path, traceback.format_exc())
        print(f"[{args.label}] [ERROR] Failed at batch {n_ok}: {error}")

    t_epoch = time.time() - t_epoch_start

    stats = summarize(batch_times)
    result_lines = [
        "=" * 70,
        f"RESULTS label={args.label}",
        f"dataset_root: {args.dataset_root}",
        f"video_backend: {args.video_backend}",
        f"batch_size: {args.batch_size}  num_workers: {args.num_workers}  prefetch_factor: {args.prefetch_factor}",
        f"total_frames_in_dataset: {len(ds)}",
        f"batches_completed: {n_ok} / {n_target} (available_per_epoch={total_batches_available})",
        f"completed_full_epoch_without_error: {error is None and n_ok == total_batches_available}",
    ]
    if error:
        result_lines.append(f"ERROR: {error}")
    if stats:
        result_lines += [
            f"avg_batch_time_s: {stats['avg']:.4f}",
            f"p50_batch_time_s: {stats['p50']:.4f}",
            f"p95_batch_time_s: {stats['p95']:.4f}",
            f"min_batch_time_s: {stats['min']:.4f}",
            f"max_batch_time_s: {stats['max']:.4f}",
            f"total_dataloader_time_s: {stats['total']:.2f}",
            f"wall_clock_epoch_time_s: {t_epoch:.2f}",
            f"samples_per_sec: {(n_ok * args.batch_size) / stats['total']:.2f}" if stats["total"] > 0 else "n/a",
        ]
    result_lines.append("=" * 70)

    summary_text = "\n".join(result_lines)
    print(summary_text)
    with open(summary_log_path, "a") as f:
        f.write("\n" + summary_text + "\n")

    if error:
        sys.exit(1)


if __name__ == "__main__":
    main()

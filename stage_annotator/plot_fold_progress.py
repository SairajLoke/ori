#!/usr/bin/env python3
"""Plot fold_progress_pipeline.py's output against REAL dataset frame indices,
interpolating progress across every frame between the fold's start/end boundary
(not just the sparse VLM-scored sample points). One plot per results file, each
showing all episodes' progress curves for that fold.

Usage (single fold):
    python plot_fold_progress.py <results.json> [--fold N] [--out out.png]

Usage (all folds you've scored so far, one plot each):
    python plot_fold_progress.py fold*_progress_results.json
    (--fold/--out are ignored with multiple files -- fold index is auto-detected
    per file, and each gets its own <input>.png next to it)

--fold: which fold index (0-5, matching stages.txt) a results file covers --
    needed to know which boundary pair (b<fold> -> b<fold+1>) to map
    timestamp_norm against. Defaults to guessing from the frame_path in the
    JSON itself (fold<N>_frames/...) rather than trusting the filename, since
    e.g. "fold1_dryrun.json" turned out to actually contain fold 0 data.
"""
import argparse
import json
import pathlib
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANNOT_PATH = pathlib.Path(__file__).parent / "annotations" / "boundaries.json"


def guess_fold_index(results: dict) -> int:
    for fr in results["frames"]:
        m = re.search(r"fold(\d+)_frames", fr["frame_path"])
        if m:
            return int(m.group(1))
    raise ValueError("couldn't find fold<N>_frames in any frame_path -- pass --fold explicitly")


def boundary_episode_frame(annot: dict, episode_index: str, b: int) -> int:
    """b0 is the implicit episode start (not stored); b1..b6 are read from boundaries.json."""
    if b == 0:
        return 0
    return annot["episodes"][episode_index][f"b{b}"]["episode_frame"]


def plot_one(results_json: pathlib.Path, fold_override: int | None, out_override: pathlib.Path | None,
             annot: dict) -> None:
    results = json.loads(results_json.read_text())
    fold = fold_override if fold_override is not None else guess_fold_index(results)

    by_episode: dict[str, list[dict]] = {}
    for fr in results["frames"]:
        by_episode.setdefault(fr["episode"], []).append(fr)

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.tab10.colors

    for i, (ep_name, frames) in enumerate(sorted(by_episode.items())):
        ep_idx = str(int(ep_name.split("_")[-1]))  # "episode_000" -> "0"
        start_ef = boundary_episode_frame(annot, ep_idx, fold)
        end_ef = boundary_episode_frame(annot, ep_idx, fold + 1)

        scored = sorted(
            [f for f in frames if f["progress_smooth"] is not None],
            key=lambda f: f["timestamp_norm"],
        )
        if len(scored) < 2:
            print(f"  {ep_name}: only {len(scored)} scored frame(s) -- can't interpolate a curve, skipping")
            continue

        real_frames = np.array([start_ef + f["timestamp_norm"] * (end_ef - start_ef) for f in scored])
        progress = np.array([f["progress_smooth"] for f in scored])

        dense_frames = np.arange(start_ef, end_ef + 1)
        dense_progress = np.interp(dense_frames, real_frames, progress)

        color = colors[i % len(colors)]
        ax.plot(dense_frames, dense_progress, "-", color=color, alpha=0.6,
                label=f"{ep_name} (interpolated, {len(scored)} real points)")
        ax.scatter(real_frames, progress, color=color, s=45, zorder=5, edgecolors="black", linewidths=0.5)

    fold_name = results.get("fold_name", f"fold{fold}")
    ax.set_xlabel("dataset frame index (episode-relative)")
    ax.set_ylabel("progress (%)")
    ax.set_title(f"Fold {fold}: {fold_name} -- progress vs. real frame index\n"
                 f"(dots = actual Gemini-scored frames, lines = linear interpolation between them)")
    ax.set_ylim(-5, 105)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = out_override or results_json.with_suffix(".png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_json", type=pathlib.Path, nargs="+",
                    help="one or more fold_progress_pipeline.py --out files")
    p.add_argument("--fold", type=int, default=None,
                    help="only meaningful with a single results_json -- ignored (auto-detected "
                         "per file) when multiple files are given")
    p.add_argument("--out", type=pathlib.Path, default=None,
                    help="only meaningful with a single results_json -- ignored (defaults to "
                         "<input>.png per file) when multiple files are given")
    args = p.parse_args()

    multi = len(args.results_json) > 1
    if multi and (args.fold is not None or args.out is not None):
        print("--fold/--out ignored: multiple result files given, auto-detecting per file")

    annot = json.loads(ANNOT_PATH.read_text())
    for rj in args.results_json:
        print(f"{rj}:")
        plot_one(rj, None if multi else args.fold, None if multi else args.out, annot)


if __name__ == "__main__":
    main()

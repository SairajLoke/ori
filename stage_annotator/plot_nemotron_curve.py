#!/usr/bin/env python3
"""Plot nemotron_stage_progress.py's --mode curve output: one figure per
results file, all episodes overlaid, progress vs. episode-relative frame
index. Mirrors plot_fold_progress.py's visual conventions (tab10 colors,
dpi=130, Agg backend) but needs no boundaries.json mapping -- unlike that
script's timestamp_norm, curve mode's frame_index is already a real,
episode-relative frame position, not a normalized [0,1] fraction.

Frames with progress=null (VLM call failed after retries -- see
nemotron_stage_progress.py's --retry_missing) show up as a genuine gap in
the line rather than being silently interpolated over, so missing data is
visually obvious rather than hidden.

Usage:
    python plot_nemotron_curve.py nemotron_fold3_curve_results.json
"""
import argparse
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_one(results_json: pathlib.Path, out_override: pathlib.Path | None) -> None:
    results = json.loads(results_json.read_text())
    by_episode = results["results"]

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.tab10.colors

    total_frames = 0
    total_missing = 0
    for i, (ep_name, rows) in enumerate(sorted(by_episode.items())):
        rows = sorted(rows, key=lambda r: r["frame_index"])
        idx = np.array([r["frame_index"] for r in rows], dtype=float)
        # NaN (not 0 or interpolation) for missing scores -- matplotlib breaks
        # the line at NaN, so a failed group shows as a visible gap, not a
        # silently-smoothed-over guess.
        progress = np.array([r["progress"] if r["progress"] is not None else np.nan for r in rows])
        n_missing = int(np.isnan(progress).sum())
        total_frames += len(rows)
        total_missing += n_missing

        color = colors[i % len(colors)]
        label = f"{ep_name} ({len(rows) - n_missing}/{len(rows)} scored)"
        ax.plot(idx, progress, "-", color=color, alpha=0.7, linewidth=1.5, label=label)
        ax.scatter(idx, progress, color=color, s=18, zorder=5, edgecolors="black", linewidths=0.3)

    fold_name = results.get("fold_name", results_json.stem)
    group_size = results.get("group_size", "?")
    ax.set_xlabel("frame index (episode-relative, within this stage's clip)")
    ax.set_ylabel("progress (%)")
    ax.set_title(
        f"{fold_name} -- per-frame progress (Nemotron, group_size={group_size})\n"
        f"dots = VLM-scored frames, gaps = failed calls not yet backfilled "
        f"({total_missing}/{total_frames} missing)"
    )
    ax.set_ylim(-5, 105)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = out_override or results_json.with_suffix(".png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  {total_frames - total_missing}/{total_frames} frames plotted -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_json", type=pathlib.Path, nargs="+")
    p.add_argument("--out", type=pathlib.Path, default=None,
                    help="only meaningful with a single results_json")
    args = p.parse_args()

    multi = len(args.results_json) > 1
    for rj in args.results_json:
        print(f"{rj}:")
        plot_one(rj, None if multi else args.out)


if __name__ == "__main__":
    main()

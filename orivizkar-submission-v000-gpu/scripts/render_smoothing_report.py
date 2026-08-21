"""Render eval_results/*.json into the results tables of SMOOTHING.md."""

from __future__ import annotations

import json
import sys
from pathlib import Path

MODES = ('none', 'anchor', 'ensemble', 'clamp', 'anchor_clamp', 'ensemble_clamp')


def table(res, strides, cols=("seam_jump_mean", "max_step", "mse_gt")):
    hdr = {"seam_jump_mean": "seam", "max_step": "max_step", "mse_gt": "MSE"}
    out = ["| mode | " + " | ".join(f"s={s} {hdr[c]}" for s in strides for c in cols) + " |",
           "|" + "---|" * (1 + len(strides) * len(cols))]
    for m in MODES:
        cells = []
        for s in strides:
            r = res.get(f"stride{s}/{m}")
            for c in cols:
                cells.append("-" if r is None else f"{r[c]:.4f}")
        out.append(f"| `{m}` | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main(results_dir: Path):
    for f in sorted(results_dir.glob("*.json")):
        d = json.loads(f.read_text())
        strides = sorted({int(k.split("/")[0][6:]) for k in d["results"]})
        print(f"\n### {f.stem}  (ckpt `{d['ckpt']}`, regime `{d['regime']}`)\n")
        print(table(d["results"], strides))
        dis = d.get("disagreement", {})
        if dis:
            print("\nRaw consecutive-chunk disagreement (no smoothing):\n")
            print("| stride | seam-row mean | seam-row max | overlap mean |")
            print("|---|---|---|---|")
            for s in sorted(dis, key=int):
                v = dis[s]
                if v:
                    print(f"| {s} | {v['seam_row_mean']:.5f} | {v['seam_row_max']:.4f} | {v['overlap_mean']:.5f} |")
        e = d.get("depth_mse", [])
        if e:
            pts = [p for p in (1, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100) if p <= len(e)]
            print("\nOpen-loop MSE by horizon depth:\n")
            print("| depth | " + " | ".join(str(p) for p in pts) + " |")
            print("|" + "---|" * (1 + len(pts)))
            print("| MSE | " + " | ".join(f"{e[p-1]:.4f}" for p in pts) + " |")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "eval_results"))

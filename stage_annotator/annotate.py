#!/usr/bin/env python3
"""Interactive stage-boundary annotator, fixed 6-stage structure (stages.txt).
Plays the RAW video file continuously (episodes are concatenated within a
file, per LeRobot v3.0 layout) and live-resolves which episode + global
dataset index the current playback position belongs to, using ONLY the
dataset's own episode metadata parquet -- never a guess. Read-only against
the dataset; writes only under stage_annotator/annotations/.

    python annotate.py --dataset_root <lerobot3.0 root> --episode 0 \
        --camera observation.images.head_left

--episode picks which video FILE to load (the one containing that episode) --
you can then scrub across every episode sharing that file, not just the one
named.

Once an episode has all 6 boundaries marked (b1-b5 = stage transitions, b6 =
end of the last stage, symmetric with the implicit b0 = start of the first),
export its frames for fold_progress_pipeline.py (non-interactive, no window
opens):

    python annotate.py --dataset_root <root> --episode 0 \
        --export_frames_root <output dir> [--export_fps 3.0]

Controls:
  <- / ->     step 1 frame
  , / .       step 10 frames
  space       play / pause
  b           mark the next boundary for the CURRENT episode immediately (no
              confirmation prompt -- press 'u' right after if that was a
              misclick). Boundaries must be marked in order, b1 before b2,
              etc. b6 marks the end of the final stage, not a transition
              into a 7th stage.
  u           undo (delete) the current episode's most recently marked boundary
  s           save all annotated episodes to annotations/boundaries.json
  q           save and quit
"""
from __future__ import annotations

import argparse
import json
import pathlib
from datetime import datetime, timezone

import cv2
import numpy as np
import pyarrow.parquet as pq

FPS = 30
ANNOT_DIR = pathlib.Path(__file__).parent / "annotations"
STAGES_PATH = pathlib.Path(__file__).parent / "stages.txt"


def load_stages() -> dict[int, str]:
    return {int(k): v for k, v in json.loads(STAGES_PATH.read_text()).items()}


def load_episode_index(dataset_root: pathlib.Path, camera: str, start_episode: int
                        ) -> tuple[pathlib.Path, list[dict]]:
    """All episodes that share the same video file as `start_episode`, sorted
    by their position in that file. Scans every episode-metadata shard (not
    just chunk-000/file-000) since a larger dataset can have more than one."""
    cols = ["episode_index", "length", f"videos/{camera}/chunk_index", f"videos/{camera}/file_index",
            f"videos/{camera}/from_timestamp", f"videos/{camera}/to_timestamp",
            "dataset_from_index", "dataset_to_index"]
    shards = sorted((dataset_root / "meta/episodes").glob("**/*.parquet"))
    rows = []
    for shard in shards:
        d = pq.read_table(shard, columns=cols).to_pydict()
        for i in range(len(d["episode_index"])):
            rows.append({k: d[k][i] for k in d})

    target = next(r for r in rows if r["episode_index"] == start_episode)
    chunk, file_ = target[f"videos/{camera}/chunk_index"], target[f"videos/{camera}/file_index"]
    same_file = [r for r in rows
                 if r[f"videos/{camera}/chunk_index"] == chunk and r[f"videos/{camera}/file_index"] == file_]
    same_file.sort(key=lambda r: r[f"videos/{camera}/from_timestamp"])
    video_path = dataset_root / "videos" / camera / f"chunk-{chunk:03d}" / f"file-{file_:03d}.mp4"
    return video_path, [
        {"episode_index": r["episode_index"],
         "from_s": r[f"videos/{camera}/from_timestamp"], "to_s": r[f"videos/{camera}/to_timestamp"],
         "dataset_from_index": r["dataset_from_index"], "length": r["length"]}
        for r in same_file
    ]


def resolve_position(file_frame: int, episodes: list[dict]) -> dict | None:
    """file_frame (0-based, within the whole video file) -> episode + indices, or
    None if between/outside known episode windows (shouldn't normally happen)."""
    t = file_frame / FPS
    for ep in episodes:
        if ep["from_s"] <= t < ep["to_s"]:
            episode_frame = file_frame - round(ep["from_s"] * FPS)
            return {"episode_index": ep["episode_index"], "episode_frame": episode_frame,
                     "time_s": round(episode_frame / FPS, 2),
                     "abs_idx": ep["dataset_from_index"] + episode_frame}
    return None


def episode_frame_to_file_frame(episode_frame: int, ep: dict) -> int:
    """Inverse of resolve_position's episode_frame computation -- episode-relative
    frame -> absolute frame within the video FILE, for seeking with cv2."""
    return round(ep["from_s"] * FPS) + episode_frame


def export_episode_frames(dataset_root: pathlib.Path, camera: str, episode_index: int,
                           export_root: pathlib.Path, export_fps: float,
                           stages: dict[int, str], episodes_data: dict) -> None:
    """Non-interactive: dump this episode's frames into fold_progress_pipeline.py's
    expected layout --

        export_root/fold<N>_frames/episode_<idx>/frame_<i>.jpg   (sampled at export_fps,
            N in 0..5, one dir per fold matching stages.txt; multiple episodes'
            exports accumulate side-by-side under the same fold<N>_frames/ so the
            pipeline can pool them for cross-episode calibration)
        export_root/anchors/episode_<idx>/b<0..6>.jpg   (the 7 stage-boundary frames:
            b0 = episode start, b1..b5 = your marked stage-transition boundaries,
            b6 = your marked end of the final stage -- NOT duplicated as separate
            "anchor_start"/"anchor_end" copies per fold, since fold N's end boundary
            IS fold N+1's start boundary; point --anchor_start/--anchor_end at the
            appropriate b<N>.jpg/b<N+1>.jpg when invoking the pipeline)
        export_root/fold_meta.json   ({"0": {"fold_name":..., "fold_description":...}, ...},
            fold_name pre-filled from stages.txt, fold_description left blank for you
            to fill in once -- never overwritten by a later export of another episode)

    Requires all 6 boundaries (b1-b6) already marked for this episode.
    """
    ep_key = str(episode_index)
    marked = episodes_data.get(ep_key, {})
    missing = [f"b{i}" for i in range(1, 7) if f"b{i}" not in marked]
    if missing:
        raise SystemExit(f"episode {episode_index} is missing boundaries {missing} -- "
                          f"finish annotating it before exporting")

    video_path, episodes = load_episode_index(dataset_root, camera, episode_index)
    ep = next(e for e in episodes if e["episode_index"] == episode_index)
    cap = cv2.VideoCapture(str(video_path))

    # b0..b6: episode-relative frame of the episode's start + the 6 marked boundaries.
    b_frames = [0] + [marked[f"b{i}"]["episode_frame"] for i in range(1, 7)]

    def read_at(episode_frame: int) -> np.ndarray:
        cap.set(cv2.CAP_PROP_POS_FRAMES, episode_frame_to_file_frame(episode_frame, ep))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"failed to read episode_frame={episode_frame} from {video_path}")
        return frame

    anchors_dir = export_root / "anchors" / f"episode_{episode_index:03d}"
    anchors_dir.mkdir(parents=True, exist_ok=True)
    for i, ef in enumerate(b_frames):
        cv2.imwrite(str(anchors_dir / f"b{i}.jpg"), read_at(ef))
    print(f"  wrote 7 anchor frames -> {anchors_dir}")

    step = max(1, round(FPS / export_fps))
    for f in range(6):
        start_ef = b_frames[f]
        end_ef = b_frames[f + 1]
        fold_dir = export_root / f"fold{f}_frames" / f"episode_{episode_index:03d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        # range(start, end+1, step) doesn't guarantee landing exactly on end_ef
        # unless (end_ef - start_ef) is a multiple of step -- up to step-1 frames
        # could be silently missing right at the boundary, which is exactly the
        # frame that matters most (it's the 0%/100% anchor reference). Force it
        # in explicitly rather than relying on the stride to land there by luck.
        frame_positions = list(range(start_ef, end_ef, step))
        if not frame_positions or frame_positions[-1] != end_ef:
            frame_positions.append(end_ef)
        for i, ef in enumerate(frame_positions):
            cv2.imwrite(str(fold_dir / f"frame_{i:04d}.jpg"), read_at(ef))
        print(f"  fold{f} ({stages[f]}): episode_frame [{start_ef}:{end_ef}] "
              f"-> {len(frame_positions)} frames @ {export_fps}fps -> {fold_dir}")

    cap.release()

    meta_path = export_root / "fold_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for f in range(6):
        meta.setdefault(str(f), {"fold_name": stages[f], "fold_description": ""})
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  fold_meta.json (fold_description left blank on first write, "
          f"never overwritten after) -> {meta_path}")


def load_annotations(dataset_root: pathlib.Path) -> dict:
    """Multi-dataset-safe: episodes are namespaced under the resolved --dataset_root
    path (not a bare episode_index), so a second dataset's episode 0 can never collide
    with the first's. Also means that after a LeRobot aggregate_datasets() merge,
    remapping is just adding a known offset per source dataset_root -- aggregate_datasets
    shifts episode_index/global index by each source's total_episodes/total_frames
    (verified against lerobot 0.6.0's aggregate.py), and writes no provenance of its own,
    so keeping our own dataset_root-keyed record is what makes that offset computable
    after the fact instead of the annotations becoming orphaned."""
    path = ANNOT_DIR / "boundaries.json"
    if not path.exists():
        return {"stages_file": "stages.txt", "datasets": {}}
    data = json.loads(path.read_text())
    if "datasets" not in data:
        # Migrate the old single-dataset flat format. It had zero provenance, so the
        # only sound attribution for existing entries is "whatever --dataset_root this
        # run was given" -- there was never any other dataset they could have come from.
        old_episodes = data.pop("episodes", {})
        old_camera = data.pop("camera", None)
        key = str(dataset_root.resolve())
        data["datasets"] = {key: {"camera": old_camera, "episodes": old_episodes}}
        print(f"[migrate] old single-dataset annotations (camera={old_camera!r}, "
              f"{len(old_episodes)} episode(s)) attributed to dataset_root={key}")
    return data


def dataset_section(data: dict, dataset_root: pathlib.Path, camera: str) -> dict:
    """This dataset_root's {"camera", "episodes"} section, created empty on first use."""
    key = str(dataset_root.resolve())
    ds = data["datasets"].setdefault(key, {"camera": None, "episodes": {}})
    if ds["camera"] not in (None, camera):
        print(f"WARNING: existing annotations for {key} were made with camera={ds['camera']!r}, "
              f"now using {camera!r} -- abs_idx/episode_frame are camera-independent so this "
              f"is fine, just noting the mismatch.")
    ds["camera"] = camera
    return ds


def save_annotations(data: dict) -> pathlib.Path:
    ANNOT_DIR.mkdir(exist_ok=True)
    path = ANNOT_DIR / "boundaries.json"
    path.write_text(json.dumps(data, indent=2))
    return path


def draw_hud(frame: np.ndarray, file_frame: int, n: int, pos: dict | None,
             next_b: int, stages: dict[int, str], playing: bool) -> np.ndarray:
    frame = frame.copy()
    cv2.putText(frame, f"file frame {file_frame}/{n-1}  {'PLAY' if playing else 'PAUSE'}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if pos is None:
        cv2.putText(frame, "outside any known episode window", (8, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.putText(frame, f"episode {pos['episode_index']}  ep_frame={pos['episode_frame']}  "
                            f"t={pos['time_s']}s  abs_idx={pos['abs_idx']}",
                    (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if next_b <= 5:
            label = f"next: b{next_b} ({stages[next_b-1]} -> {stages[next_b]})"
        elif next_b == 6:
            label = f"next: b6 (end of {stages[5]} / episode end)"
        else:
            label = "all 6 boundaries marked for this episode"
        cv2.putText(frame, label, (8, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return frame


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=pathlib.Path, required=True)
    p.add_argument("--episode", type=int, required=True, help="picks which video FILE to load")
    p.add_argument("--camera", default="observation.images.head_left")
    p.add_argument("--export_frames_root", type=pathlib.Path, default=None,
                    help="if set, non-interactively export --episode's frames into "
                         "fold_progress_pipeline.py's expected layout under this dir, "
                         "then exit (no annotation window opens). Requires all 5 "
                         "boundaries already marked for --episode.")
    p.add_argument("--export_fps", type=float, default=3.0,
                    help="frame-sampling rate for --export_frames_root (default 3fps -- "
                         "a rough progress trace, not every native 30fps frame).")
    args = p.parse_args()

    stages = load_stages()

    if args.export_frames_root is not None:
        data = load_annotations(args.dataset_root)
        ds = dataset_section(data, args.dataset_root, args.camera)
        export_episode_frames(args.dataset_root, args.camera, args.episode,
                              args.export_frames_root, args.export_fps, stages, ds["episodes"])
        return 0

    video_path, episodes = load_episode_index(args.dataset_root, args.camera, args.episode)
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{video_path}\n{n} frames, {len(episodes)} episode(s) in this file: "
          f"{[e['episode_index'] for e in episodes]}")

    data = load_annotations(args.dataset_root)
    ds = dataset_section(data, args.dataset_root, args.camera)
    print(__doc__.split("Controls:")[1])

    # Resume where this episode's annotation left off, not the start of the FILE
    # (which is only the start of episode 0 within it -- e.g. --episode 2 previously
    # always opened on whatever episode 0's frames happen to be).
    ep = next(e for e in episodes if e["episode_index"] == args.episode)
    marked = ds["episodes"].get(str(args.episode), {})
    if marked:
        last_b = max(int(k[1:]) for k in marked)
        start_ef = marked[f"b{last_b}"]["episode_frame"]
        print(f"resuming episode {args.episode} from b{last_b} (episode_frame={start_ef})")
    else:
        start_ef = 0
        print(f"episode {args.episode} has no boundaries marked yet -- starting from its beginning")
    idx, playing = episode_frame_to_file_frame(start_ef, ep), False
    win = f"stage annotator"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def current_pos() -> dict | None:
        return resolve_position(idx, episodes)

    def next_boundary(ep_idx: int) -> int:
        marked = ds["episodes"].get(str(ep_idx), {})
        for b in range(1, 7):
            if f"b{b}" not in marked:
                return b
        return 7  # all marked

    def show() -> np.ndarray | None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            return None
        pos = current_pos()
        nb = next_boundary(pos["episode_index"]) if pos else 7
        return draw_hud(frame, idx, n, pos, nb, stages, playing)

    frame = show()
    while True:
        if frame is not None:
            cv2.imshow(win, frame)
        key = cv2.waitKey(33 if playing else 0) & 0xFF

        if playing:
            idx = min(idx + 1, n - 1)
            if idx == n - 1:
                playing = False
            frame = show()
            if key == 255:
                continue

        if key == ord("q"):
            save_annotations(data)
            break
        elif key == ord("s"):
            path = save_annotations(data)
            print(f"saved -> {path}")
        elif key == 32:
            playing = not playing
        elif key in (81, ord("h")):
            idx = max(0, idx - 1); frame = show()
        elif key in (83, ord("l")):
            idx = min(n - 1, idx + 1); frame = show()
        elif key == ord(","):
            idx = max(0, idx - 10); frame = show()
        elif key == ord("."):
            idx = min(n - 1, idx + 10); frame = show()
        elif key == ord("b"):
            pos = current_pos()
            if pos is None:
                print("not inside a known episode window -- can't mark a boundary here")
                continue
            b = next_boundary(pos["episode_index"])
            if b > 6:
                print(f"episode {pos['episode_index']} already has all 6 boundaries marked")
                continue
            if b <= 5:
                prompt_label = f"b{b} ({stages[b-1]} -> {stages[b]})"
                to_stage = b
            else:
                prompt_label = f"b6 (end of {stages[5]} / episode end)"
                to_stage = None
            ep_key = str(pos["episode_index"])
            ds["episodes"].setdefault(ep_key, {})[f"b{b}"] = {
                "abs_idx": pos["abs_idx"], "episode_frame": pos["episode_frame"],
                "time_s": pos["time_s"], "from_stage": b - 1, "to_stage": to_stage,
                "annotated_at": datetime.now(timezone.utc).isoformat(),
            }
            print(f"  marked {prompt_label} at episode_frame={pos['episode_frame']} "
                  f"(abs_idx={pos['abs_idx']}) for episode {pos['episode_index']} -- "
                  f"press 'u' to undo")
            frame = show()
        elif key == ord("u"):
            pos = current_pos()
            if pos is None:
                continue
            ep_key = str(pos["episode_index"])
            marked = ds["episodes"].get(ep_key, {})
            if not marked:
                print(f"no boundaries marked yet for episode {pos['episode_index']}")
                continue
            last_b = max(int(k[1:]) for k in marked)
            del ds["episodes"][ep_key][f"b{last_b}"]
            print(f"  invalidated b{last_b} for episode {pos['episode_index']}")
            frame = show()
            frame = show()

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# (.venv) (base) sai@sai:~/Desktop/ORI/ori/stage_annotator$ 
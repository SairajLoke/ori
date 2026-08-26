#!/usr/bin/env python3
"""Paper segmentation via SAM2 video mask propagation: prompt once (a point on
the paper in frame 0), propagate across the whole clip, save a boolean mask
per frame. Built to feed CoTracker's query grid -- restrict points to
mask==True instead of a naive whole-frame grid (today's run put ~400 points
across the frame and got ~1 on the paper, since paper is 0.14-1.05% of frame
per plans.md's own measurement).

GPU-only in practice (Colab): SAM2's video predictor keeps a per-frame memory
bank for propagation, architecturally the same class of memory cost as
CoTracker's online mode -- not worth fighting on this box's 15GB CPU RAM
given today's lesson on that exact failure mode.

Setup (once, e.g. in a Colab cell):
    !git clone https://github.com/facebookresearch/sam2.git
    %cd sam2 && pip install -q -e . && cd checkpoints && ./download_ckpts.sh && cd ..

Usage (manual point -- pick by eye from --dump_first_frame):
    python run_sam2_paper_mask.py --video <path> --point X Y \
        [--seconds 90] [--start_seconds 60] [--out_name ep0_head_left]

Usage (automatic -- Grounding DINO detects "a paper" for the box prompt and
"a gripper" for negative refinement points, zero manual coordinates, needed for
anything that has to run unattended across many episodes):
    python run_sam2_paper_mask.py --video <path> --auto_prompt [--seconds 90]

Usage (full scene -- also tracks left_hand/left_arm/right_hand/right_arm/table
as separate SAM2 objects in the same propagation pass, all via Grounding
DINO, still zero manual coordinates):
    python run_sam2_paper_mask.py --video <path> --auto_prompt --full_scene [--seconds 90]

Either --point or --auto_prompt is required -- SAM2 itself has no notion of
"paper", it only propagates a mask from whatever prompt it's given.
"""
from __future__ import annotations

import argparse
import pathlib
import tempfile

import cv2
import numpy as np
import torch

FPS = 30
OUT_DIR = pathlib.Path(__file__).parent / "results"

# checkpoint filename / config filename per size, keyed the same way so --model_size picks both
SAM2_VARIANTS = {
    "tiny": ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "small": ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "base_plus": ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "large": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}


def read_clip(path: pathlib.Path, start_seconds: float, seconds: float | None) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if start_seconds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(start_seconds * FPS))
    n = round(seconds * FPS) if seconds is not None else None
    frames = []
    while n is None or len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(frames) % 500 == 0:
            print(f"    read {len(frames)}{f'/{n}' if n else ''} frames", flush=True)
    cap.release()
    print(f"    read {len(frames)} frames total", flush=True)
    return np.stack(frames)


def _split_left_right(boxes: list[tuple[list[float], float]], frame_width: int,
                       left_name: str, right_name: str) -> dict[str, list[float]]:
    """Keep at most the 2 highest-confidence boxes, assign by x-center (left half of
    frame = left_*) -- matches this repo's left/right convention everywhere else."""
    top = sorted(boxes, key=lambda t: t[1], reverse=True)[:2]
    out = {}
    for box, _ in top:
        cx = (box[0] + box[2]) / 2
        out[left_name if cx < frame_width / 2 else right_name] = box
    return out


def detect_objects(frame: np.ndarray, box_threshold: float, text_threshold: float,
                    full_scene: bool) -> tuple[dict[str, list[float]], np.ndarray]:
    """Grounding DINO on frame 0. Always detects "a paper" (box prompt).
    With full_scene=True, also detects "a gripper" and "an arm" (each split
    left/right by box x-center) and "a table" -- every one becomes its own
    tracked SAM2 object: origami_paper, left_hand, left_arm, right_hand,
    right_arm, table. Gripper centers additionally serve as negative points
    that refine origami_paper's own boundary away from the gripper, independent
    of whether grippers are also being tracked as their own object.

    Verified against the real HF transformers API (AutoModelForZeroShot
    ObjectDetection + post_process_grounded_object_detection), not guessed
    from memory -- docs/transformers/model_doc/grounding-dino.

    Returns ({name: box_xyxy}, negative_points[N,2]) -- negative_points may be empty.
    """
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

    image = Image.fromarray(frame)
    # "a gripper"/"an arm" (not "a hand"/"a robot arm") -- verified by direct comparison that
    # this phrasing keeps hand and arm as separate detections; "a hand"/"a robot arm" merged
    # into one label per box ('a hand a robot arm'), which silently gave both objects identical
    # prompt coordinates and identical masks the whole clip.
    queries = ["a paper", "a gripper"] + (["an arm", "a table"] if full_scene else [])
    inputs = processor(images=image, text=[queries], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=box_threshold, text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    print(results['labels'])
    
    
    paper_box, paper_score = None, -1.0
    hand_centers, hand_boxes, arm_boxes, table_box, table_score = [], [], [], None, -1.0
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        box = box.tolist()
        print(f"    detected {label!r} score={score.item():.2f} box={[round(x,1) for x in box]}")
        # Grounding DINO can return a label spanning >1 adjacent text query when it isn't sure
        # of the phrase boundary -- happened with "a hand"/"a robot arm" ('a hand a robot arm'
        # for ONE box covering the whole limb), which silently gave the hand and arm objects
        # identical prompt coordinates -> identical masks the whole clip (caught by left_hand's
        # stats being pixel-identical to left_arm's). "a gripper"/"an arm" avoids the merge in
        # practice (verified by direct comparison), but keep the has_x/has_y split as a safety
        # net: a merged label still counts as arm only, since a label saying "gripper" WITHOUT
        # "arm" is what becomes its own tracked hand object. Paper's negative-point refinement
        # is unaffected either way -- that fires on any "gripper" mention, merged or not.
        has_hand, has_arm = "gripper" in label, "arm" in label
        if "paper" in label and score.item() > paper_score:
            paper_box, paper_score = box, score.item()
        if has_hand:
            hand_centers.append([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        if has_hand and not has_arm:
            hand_boxes.append((box, score.item()))
        if has_arm:
            arm_boxes.append((box, score.item()))
        if "table" in label and score.item() > table_score:
            table_box, table_score = box, score.item()

    if paper_box is None:
        raise SystemExit("Grounding DINO found no 'paper' box above threshold -- "
                          "lower --box_threshold or fall back to --point")

    objects = {"origami_paper": paper_box}
    if full_scene:
        w = frame.shape[1]
        objects.update(_split_left_right(hand_boxes, w, "left_hand", "right_hand"))
        objects.update(_split_left_right(arm_boxes, w, "left_arm", "right_arm"))
        if table_box is not None:
            objects["table"] = table_box
    return objects, np.array(hand_centers, dtype=np.float32)


# BGR, one per object name
# BGR, 6 maximally-distinct primary/secondary colors -- one per tracked object
OBJECT_COLORS = {"origami_paper": (0, 255, 0), "left_hand": (255, 0, 255),
                  "left_arm": (255, 0, 0), "right_hand": (255, 255, 0),
                  "right_arm": (0, 0, 255), "table": (0, 255, 255)}


def overlay_mask_video(frames: np.ndarray, masks: dict[str, np.ndarray], out_path: pathlib.Path) -> None:
    """masks: {name: [T,H,W] bool}. Each object tinted its own color, semi-transparent."""
    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for t in range(frames.shape[0]):
        frame = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR).copy()
        overlay = frame.copy()
        for name, m in masks.items():
            overlay[m[t]] = OBJECT_COLORS.get(name, (0, 255, 0))
        blended = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        writer.write(blended)
    writer.release()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=pathlib.Path, required=True)
    p.add_argument("--point", type=int, nargs=2, metavar=("X", "Y"),
                    help="pixel coords on the paper in the clip's first frame")
    p.add_argument("--start_seconds", type=float, default=0.0)
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--out_name", default=None)
    p.add_argument("--dump_first_frame", action="store_true",
                    help="save results/<out_name>_frame0.png and exit -- use this first to pick --point")
    p.add_argument("--sam2_root", type=pathlib.Path, default=pathlib.Path("/content/sam2"),
                    help="dir the sam2 repo was cloned into (contains checkpoints/ and sam2/configs/)")
    p.add_argument("--model_size", choices=list(SAM2_VARIANTS), default="large")
    p.add_argument("--auto_prompt", action="store_true",
                    help="Grounding DINO picks the paper box + negative hand points -- no --point needed")
    p.add_argument("--full_scene", action="store_true",
                    help="also track left_hand/left_arm/right_hand/right_arm/table as separate "
                         "SAM2 objects (requires --auto_prompt)")
    p.add_argument("--box_threshold", type=float, default=0.3)
    p.add_argument("--text_threshold", type=float, default=0.25)
    args = p.parse_args()
    if args.full_scene and not args.auto_prompt:
        raise SystemExit("--full_scene requires --auto_prompt (arm/table boxes are detected, not clicked)")

    OUT_DIR.mkdir(exist_ok=True)
    name = args.out_name or args.video.stem
    frames = read_clip(args.video, args.start_seconds, args.seconds if args.dump_first_frame is False else 1)

    if args.dump_first_frame:
        out = OUT_DIR / f"{name}_frame0.png"
        cv2.imwrite(str(out), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
        print(f"first frame -> {out}; pick a paper pixel (x,y) from it and pass --point X Y")
        return 0

    if not args.auto_prompt and args.point is None:
        raise SystemExit("--point X Y is required, or pass --auto_prompt (or run --dump_first_frame first)")

    objects = negative_points = None
    if args.auto_prompt:
        print("running Grounding DINO on frame 0 for an automatic prompt...")
        objects, negative_points = detect_objects(frames[0], args.box_threshold, args.text_threshold, args.full_scene)
        print(f"    objects={list(objects)}, {len(negative_points)} negative hand point(s) (paper only)")

    import sys
    sys.path.insert(0, str(args.sam2_root))
    from sam2.build_sam import build_sam2_video_predictor  # local import: only needed past this point

    ckpt_name, config = SAM2_VARIANTS[args.model_size]
    checkpoint = args.sam2_root / "checkpoints" / ckpt_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"model: {args.model_size} -- {checkpoint}")
    predictor = build_sam2_video_predictor(config, str(checkpoint), device=device)

    # init_state only accepts an mp4 path or a directory of "<N>.jpg" frames (verified against
    # sam2/utils/misc.py:load_video_frames -- a raw frame array is NOT supported and raises
    # NotImplementedError). Dump our already-sliced clip there instead of pointing at the raw
    # video file, so --start_seconds/--seconds still apply.
    jpg_dir = tempfile.mkdtemp(prefix="sam2_frames_")
    for i, frame in enumerate(frames):
        cv2.imwrite(f"{jpg_dir}/{i:05d}.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"    wrote {len(frames)} JPEG frames -> {jpg_dir}")

    # init_state loads the WHOLE clip as one upfront [T,3,1024,1024] float32 tensor
    # (sam2/utils/misc.py:load_video_frames_from_jpg_images) -- not chunked, unlike
    # CoTracker's online mode. Independent of --model_size (that's network weights only).
    est_gb = len(frames) * 3 * 1024 * 1024 * 4 / 2**30
    print(f"    est. frame-tensor memory: {est_gb:.1f} GB (T={len(frames)} @ 1024x1024 float32)")
    if est_gb > 4:
        print(f"    WARNING: likely to OOM on this box -- try a shorter --seconds", flush=True)
    if args.full_scene:
        print(f"    --full_scene: tracking {len(objects)} objects adds per-object memory-bank "
              f"cost on top of the estimate above (no verified formula for that part -- "
              f"if it OOMs, cut --seconds further)", flush=True)

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32):
        state = predictor.init_state(video_path=jpg_dir)
        if args.auto_prompt:
            # All objects must be added before the first propagate_in_video call --
            # sam2_video_predictor.py explicitly forbids adding one after tracking starts.
            # Object ids assigned in `objects` dict order (insertion order, Python 3.7+).
            obj_ids = {n: i + 1 for i, n in enumerate(objects)}
            for obj_name, box in objects.items():
                # negative hand-center points only refine the paper's boundary away from the
                # gripper -- both points and box are additive in add_new_points_or_box, not
                # mutually exclusive (verified against the function's own signature).
                kwargs = dict(box=np.array(box, dtype=np.float32))
                if obj_name == "origami_paper" and len(negative_points):
                    kwargs["points"] = negative_points
                    kwargs["labels"] = np.zeros(len(negative_points), dtype=np.int32)  # 0 = negative
                predictor.add_new_points_or_box(state, frame_idx=0, obj_id=obj_ids[obj_name], **kwargs)
        else:
            obj_ids = {"paper": 1}
            predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=1,
                points=np.array([args.point], dtype=np.float32),
                labels=np.array([1], dtype=np.int32),  # 1 = foreground click
            )
        id_to_name = {v: k for k, v in obj_ids.items()}
        masks = {n: np.zeros(frames.shape[:3], dtype=bool) for n in obj_ids}  # {name: [T,H,W]}
        for frame_idx, frame_obj_ids, mask_logits in predictor.propagate_in_video(state):
            for i, oid in enumerate(frame_obj_ids):
                masks[id_to_name[oid]][frame_idx] = (mask_logits[i] > 0).cpu().numpy().squeeze()
            if frame_idx % 500 == 0:
                print(f"    propagated {frame_idx}/{frames.shape[0]}", flush=True)

    for obj_name, m in masks.items():
        pct = m.mean(axis=(1, 2))
        print(f"{obj_name} mask covers {pct.mean()*100:.2f}% of frame on average "
              f"(min {pct.min()*100:.2f}%, max {pct.max()*100:.2f}%)")

    out_path = OUT_DIR / f"{name}_papermask_overlay.mp4"
    overlay_mask_video(frames, masks, out_path)
    np.savez_compressed(OUT_DIR / f"{name}_papermask.npz", **masks)
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

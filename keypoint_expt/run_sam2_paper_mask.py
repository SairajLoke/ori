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
"a hand" for negative refinement points, zero manual coordinates, needed for
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
import bisect
import pathlib
import shutil
import tempfile

import cv2
import numpy as np
import torch
from transformers.image_transforms import center_to_corners_format

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


# [CLS], [SEP], '.' -- verified against the actual grounding-dino-tiny tokenizer
# (bert-base-uncased under the hood): tokenizer.cls_token_id == 101, sep_token_id == 102,
# convert_tokens_to_ids('.') == 1012.
_SEP_TOKEN_IDS = {101, 102, 1012}


def _phrase_from_posmap_confined(posmap_row: torch.Tensor, input_ids_row: torch.Tensor,
                                  max_idx: torch.Tensor, tokenizer) -> str:
    """One box's label text, confined to the single query phrase (the span between the
    nearest separator tokens on each side) containing that box's single highest-confidence
    token. Port of the original IDEA-Research/GroundingDINO repo's
    groundingdino.util.inference.predict(..., remove_combined=True) fix for
    https://github.com/IDEA-Research/GroundingDINO/issues/85 -- confirmed (by reading the
    installed transformers source) that the HF port's own post_process_grounded_object_detection
    / get_phrases_from_posmap has NO equivalent: it always extracts every above-text_threshold
    token across the WHOLE caption, which is exactly what lets adjacent query phrases merge
    into one label (observed here across 4 different phrasings on the same scene)."""
    sep_idx = [i for i, tok in enumerate(input_ids_row.tolist()) if tok in _SEP_TOKEN_IDS]
    insert_idx = bisect.bisect_left(sep_idx, max_idx.item())
    left_idx, right_idx = sep_idx[insert_idx - 1], sep_idx[insert_idx]
    posmap_row = posmap_row.clone()
    posmap_row[: left_idx + 1] = False
    posmap_row[right_idx:] = False
    token_ids = input_ids_row[posmap_row.nonzero(as_tuple=True)[0]]
    return tokenizer.decode(token_ids).replace(".", "").strip()


def _post_process_remove_combined(outputs, inputs, processor, box_threshold: float,
                                   text_threshold: float, target_size: tuple[int, int]) -> dict:
    """Same box/score computation as GroundingDinoProcessor.post_process_grounded_object_detection
    (single image only), but with remove_combined-style confined label extraction in place of
    HF's whole-caption extraction -- see _phrase_from_posmap_confined."""
    probs = torch.sigmoid(outputs.logits[0])            # (num_queries, 256)
    scores = torch.max(probs, dim=-1)[0]                 # (num_queries,)
    boxes = center_to_corners_format(outputs.pred_boxes[0])
    img_h, img_w = target_size
    boxes = boxes * torch.tensor([img_w, img_h, img_w, img_h], dtype=boxes.dtype)

    keep = scores > box_threshold
    scores, boxes, probs = scores[keep], boxes[keep], probs[keep]
    input_ids_row = inputs.input_ids[0]

    labels = [
        _phrase_from_posmap_confined(prob_row > text_threshold, input_ids_row,
                                      prob_row.argmax(), processor.tokenizer)
        for prob_row in probs
    ]
    return {"scores": scores, "boxes": boxes, "labels": labels}


def detect_objects(frame: np.ndarray, box_threshold: float, text_threshold: float,
                    full_scene: bool) -> tuple[dict[str, list[float]], np.ndarray]:
    """Grounding DINO on frame 0. Always detects "a paper" (box prompt).
    With full_scene=True, also detects "a hand" and "a robot arm" (each split
    left/right by box x-center) and "a table" -- every one becomes its own
    tracked SAM2 object: origami_paper, left_hand, left_arm, right_hand,
    right_arm, table. Hand centers additionally serve as negative points that
    refine origami_paper's own boundary away from the hand, independent of
    whether hands are also being tracked as their own object.

    Label extraction uses _post_process_remove_combined (see its docstring),
    a manual port of the original GroundingDINO repo's remove_combined=True fix
    for https://github.com/IDEA-Research/GroundingDINO/issues/85 -- the installed
    HF transformers version has no equivalent and would merge adjacent query
    phrases into one label (confirmed here across 4 different phrasings before
    landing on this fix, e.g. 'a gripper an arm' for a single box).

    Returns ({name: box_xyxy}, negative_points[N,2]) -- negative_points may be empty.
    """
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

    image = Image.fromarray(frame)
    # Plain wording is fine now -- _post_process_remove_combined confines each box's label
    # to the query segment around its single peak-confidence token (see that function's
    # docstring), so adjacent phrases can no longer merge regardless of how they're worded.
    if full_scene:
        queries = ["a paper", "a robot arm", "a table"]
    else:
        queries = ["a paper", "a hand"]

    inputs = processor(images=image, text=[queries], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    results = _post_process_remove_combined(
        outputs, inputs, processor, box_threshold, text_threshold,
        target_size=image.size[::-1],
    )

    print(results['labels'])

    paper_box, paper_score = None, -1.0
    hand_centers, hand_boxes, arm_boxes, table_box, table_score = [], [], [], None, -1.0
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        box = box.tolist()
        print(f"    detected {label!r} score={score.item():.2f} box={[round(x,1) for x in box]}")
        # With _post_process_remove_combined, each label is confined to a single query
        # phrase, so "hand" and "arm" no longer both appear in the same label -- these are
        # now just plain independent checks, not a merge-safety net.
        has_hand = "hand" in label
        has_arm = "arm" in label
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


def _box_from_mask(mask: np.ndarray) -> list[float] | None:
    """mask: [H,W] bool, one frame. [x0,y0,x1,y1] bounding box of the True pixels,
    or None if the mask is empty (object not visible / track lost this frame -- e.g.
    moved out of frame). Used to hand a tracked object's box forward into the next
    chunk's prompt, so a fresh init_state can pick up where the last one left off."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


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
    p.add_argument("--chunk_frames", type=int, default=300,
                    help="process this many frames at a time (default 300 = 10s @ 30fps, "
                         "~3.75GB peak SAM2 tensor). SAM2's init_state loads an ENTIRE clip "
                         "as one [T,3,1024,1024] float32 tensor upfront regardless of "
                         "--model_size/CPU-vs-GPU offload -- chunking is what actually bounds "
                         "memory for a long --seconds, not model size. Object boxes carry "
                         "forward between chunks (from the previous chunk's last propagated "
                         "mask); with --auto_prompt, Grounding DINO re-detects each chunk too "
                         "and is used as a fallback if a track was lost. Trade-off: each new "
                         "chunk's memory bank starts cold (seeded only from the box prompt, no "
                         "carried-over temporal context -- SAM2 has no cross-init_state "
                         "continuity), so the first 1-2 frames of every chunk are visibly "
                         "noisier than mid-chunk frames before the memory bank rebuilds "
                         "(verified by eye on a real clip). Bigger chunks = fewer boundaries = "
                         "fewer of these dips, at the cost of more peak memory per chunk.")
    args = p.parse_args()
    if args.full_scene and not args.auto_prompt:
        raise SystemExit("--full_scene requires --auto_prompt (arm/table boxes are detected, not clicked)")

    OUT_DIR.mkdir(exist_ok=True)
    name = args.out_name or args.video.stem

    if args.dump_first_frame:
        frame0 = read_clip(args.video, args.start_seconds, 1)
        out = OUT_DIR / f"{name}_frame0.png"
        cv2.imwrite(str(out), cv2.cvtColor(frame0[0], cv2.COLOR_RGB2BGR))
        print(f"first frame -> {out}; pick a paper pixel (x,y) from it and pass --point X Y")
        return 0

    if not args.auto_prompt and args.point is None:
        raise SystemExit("--point X Y is required, or pass --auto_prompt (or run --dump_first_frame first)")

    import sys
    sys.path.insert(0, str(args.sam2_root))
    from sam2.build_sam import build_sam2_video_predictor  # local import: only needed past this point

    ckpt_name, config = SAM2_VARIANTS[args.model_size]
    checkpoint = args.sam2_root / "checkpoints" / ckpt_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"model: {args.model_size} -- {checkpoint}")
    predictor = build_sam2_video_predictor(config, str(checkpoint), device=device)

    est_gb = args.chunk_frames * 3 * 1024 * 1024 * 4 / 2**30
    print(f"    chunk_frames={args.chunk_frames} -> est. peak frame-tensor memory: {est_gb:.1f} "
          f"GB/chunk (T=chunk_frames @ 1024x1024 float32, independent of --model_size)")
    if est_gb > 4:
        print(f"    WARNING: likely to OOM even per-chunk -- lower --chunk_frames", flush=True)

    # Process in bounded chunks so only ONE chunk's frames/SAM2 tensor is ever resident at
    # once (init_state loads a whole chunk upfront -- see --chunk_frames help above). Each
    # chunk is a fresh init_state/propagate_in_video call (SAM2 has no cross-call memory
    # continuity); object boxes are handed forward via _box_from_mask on the previous
    # chunk's last frame so tracking doesn't restart from scratch each chunk.
    total_requested = round(args.seconds * FPS) if args.seconds is not None else None
    all_frames: list[np.ndarray] = []
    all_masks: dict[str, list[np.ndarray]] = {}
    last_boxes: dict[str, list[float] | None] = {}
    obj_order: list[str] | None = None  # fixed tracked-object-name order, set at chunk 0
    frames_done = 0
    chunk_idx = 0

    while total_requested is None or frames_done < total_requested:
        chunk_seconds = (args.chunk_frames / FPS if total_requested is None
                         else min(args.chunk_frames, total_requested - frames_done) / FPS)
        chunk_start = args.start_seconds + frames_done / FPS
        print(f"--- chunk {chunk_idx}: frames [{frames_done}:{frames_done + round(chunk_seconds * FPS)}) "
              f"(t={chunk_start:.1f}s) ---", flush=True)
        chunk = read_clip(args.video, chunk_start, chunk_seconds)
        if len(chunk) == 0:
            break  # ran off the end of the video

        objects: dict[str, list[float]] = {}
        negative_points = np.zeros((0, 2), dtype=np.float32)
        if args.auto_prompt:
            print("    running Grounding DINO on this chunk's frame 0...")
            try:
                detected, negative_points = detect_objects(chunk[0], args.box_threshold,
                                                            args.text_threshold, args.full_scene)
            except SystemExit as e:
                if last_boxes.get("origami_paper") is None:
                    raise  # no continuity box either -- genuinely nothing to prompt paper with
                print(f"    WARNING: {e} -- continuing with the previous chunk's tracked box(es) only")
                detected = {}
            if obj_order is None:
                obj_order = list(detected)
            objects = dict(detected)
            for onm, box in last_boxes.items():
                if box is not None:
                    objects[onm] = box  # prefer continuity over a fresh (possibly drifted) redetection
            missing = [n for n in obj_order if n not in objects]
            if missing:
                print(f"    WARNING: no box available this chunk for {missing} -- untracked "
                      f"(mask stays empty) until redetected in a later chunk")
            print(f"    objects={list(objects)}, {len(negative_points)} negative hand point(s) (paper only)")
        else:
            obj_order = obj_order or ["paper"]
            if chunk_idx > 0 and last_boxes.get("paper") is None:
                raise SystemExit(f"chunk {chunk_idx}: paper track lost and no auto-detector to "
                                 f"fall back on in manual --point mode -- use --auto_prompt for "
                                 f"automatic re-detection between chunks")

        jpg_dir = tempfile.mkdtemp(prefix="sam2_frames_")
        try:
            for i, frame in enumerate(chunk):
                cv2.imwrite(f"{jpg_dir}/{i:05d}.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32):
                state = predictor.init_state(video_path=jpg_dir)
                # All objects must be added before the first propagate_in_video call --
                # sam2_video_predictor.py explicitly forbids adding one after tracking starts.
                obj_ids = {n: i + 1 for i, n in enumerate(obj_order)}
                if args.auto_prompt:
                    for obj_name in obj_order:
                        if obj_name not in objects:
                            continue  # no box this chunk (see WARNING above)
                        # negative hand-center points only refine the paper's boundary away from
                        # the gripper -- both points and box are additive in add_new_points_or_box,
                        # not mutually exclusive (verified against the function's own signature).
                        kwargs = dict(box=np.array(objects[obj_name], dtype=np.float32))
                        if obj_name == "origami_paper" and len(negative_points):
                            kwargs["points"] = negative_points
                            kwargs["labels"] = np.zeros(len(negative_points), dtype=np.int32)  # 0 = negative
                        predictor.add_new_points_or_box(state, frame_idx=0, obj_id=obj_ids[obj_name], **kwargs)
                elif chunk_idx == 0:
                    predictor.add_new_points_or_box(
                        state, frame_idx=0, obj_id=obj_ids["paper"],
                        points=np.array([args.point], dtype=np.float32),
                        labels=np.array([1], dtype=np.int32),  # 1 = foreground click
                    )
                else:
                    predictor.add_new_points_or_box(
                        state, frame_idx=0, obj_id=obj_ids["paper"],
                        box=np.array(last_boxes["paper"], dtype=np.float32))

                id_to_name = {v: k for k, v in obj_ids.items()}
                chunk_masks = {n: np.zeros(chunk.shape[:3], dtype=bool) for n in obj_order}
                for frame_idx, frame_obj_ids, mask_logits in predictor.propagate_in_video(state):
                    for i, oid in enumerate(frame_obj_ids):
                        chunk_masks[id_to_name[oid]][frame_idx] = (mask_logits[i] > 0).cpu().numpy().squeeze()
                    if frame_idx % 100 == 0:
                        print(f"    propagated {frame_idx}/{chunk.shape[0]}", flush=True)

            del state
            if device == "cuda":
                torch.cuda.empty_cache()
        finally:
            shutil.rmtree(jpg_dir, ignore_errors=True)

        for onm in obj_order:
            all_masks.setdefault(onm, []).append(chunk_masks[onm])
            last_boxes[onm] = _box_from_mask(chunk_masks[onm][-1])
        all_frames.append(chunk)

        got = len(chunk)
        frames_done += got
        chunk_idx += 1
        if got < round(chunk_seconds * FPS):
            break  # short read -> end of video

    frames = np.concatenate(all_frames, axis=0)
    masks = {onm: np.concatenate(parts, axis=0) for onm, parts in all_masks.items()}

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

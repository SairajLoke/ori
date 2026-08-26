"""
fold_progress_pipeline.py

End-to-end pipeline for estimating nonlinear, cross-episode-calibrated progress
within a single origami fold stage using a VLM (Gemini) with few-shot anchors.

Pipeline:
  1. Load frames for a given fold stage across multiple episodes
  2. Call Gemini with anchor images (0% / 100%) + shuffled frame batch
  3. Parse JSON progress estimates
  4. Smooth per-episode progress curves with isotonic regression (preserves nonlinearity,
     enforces monotonicity, removes VLM noise)
  5. Save results + basic cross-episode consistency diagnostics

Usage:
    export GEMINI_API_KEY=your_key_here   # get one free at https://aistudio.google.com/apikey
    python fold_progress_pipeline.py --fold_dir /path/to/fold3_frames --fold_name "wing valley fold" \
        --anchor_start /path/to/ep0_start.jpg /path/to/ep1_start.jpg /path/to/ep2_start.jpg \
        --anchor_end   /path/to/ep0_end.jpg   /path/to/ep1_end.jpg   /path/to/ep2_end.jpg

    --anchor_start/--anchor_end each take ONE OR MORE image paths (shell-glob friendly, e.g.
    auto_progress/anchors/episode_*/b0.jpg). Anchoring against a single episode's 0%/100%
    frame calibrates every OTHER episode's score against that one episode's specific visual
    idiosyncrasies; pooling one example per episode gives the model several concrete instances
    to generalize the "0%"/"100%" concept from instead of pattern-matching to one clip.

    Dry run first (only processes 1 batch, cheap way to sanity-check before spending quota):
    python fold_progress_pipeline.py --fold_dir /path/to/fold3_frames --fold_name "wing valley fold" \
        --anchor_start /path/to/ep0_start.jpg --anchor_end /path/to/ep0_end.jpg --max_batches 1

Expected frame directory layout:
    fold_dir/
        episode_001/
            frame_0000.jpg
            frame_0001.jpg
            ...
        episode_002/
            ...

Notes on Gemini free tier (as of 2026):
  - Flash and Flash-Lite are free (Pro is paid-only now). Default model here is
    gemini-3.6-flash (gemini-2.5-flash was retired for new API keys).
  - ACTUAL observed free-tier limit for gemini-3.6-flash specifically:
    GenerateRequestsPerDayPerProjectPerModel-FreeTier = 20 requests/day (confirmed via
    a real 429 RESOURCE_EXHAUSTED response, quotaValue: '20'). The "~1,500 requests/day"
    previously claimed here was wrong -- untested, carried over from stale/generic docs,
    and not specific to this model. Quota is tracked per (project, model), so a
    different model (e.g. gemini-3.5-flash-lite) likely has its own separate, unrelated
    pool -- don't assume it shares this one's remaining budget either way without
    checking. At BATCH_SIZE=12, this caps you at ~1-4 folds/day depending on how many
    episodes/sample_per_episode you're running.
  - Free-tier inputs/outputs may be used by Google to improve their models — avoid
    the free tier if your footage is sensitive.
  - BATCH_SIZE is kept modest and a small delay is added between calls to stay
    comfortably under the RPM cap; raise/lower depending on your quota tier. This does
    NOT help with the per-day cap above, only the per-minute one.

Requires: pip install google-genai scikit-learn numpy --break-system-packages
"""

import os
import io
import json
import time
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np
from sklearn.isotonic import IsotonicRegression

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None


MODEL = "gemini-3.6-flash"   # free-tier vision-capable Gemini model (gemini-2.5-flash was
                              # retired for new API keys -- 404 pointed here directly)
BATCH_SIZE = 12               # frames per API call, across all episodes combined
MAX_RETRIES = 3
SECONDS_BETWEEN_CALLS = 4.5   # ~13 RPM, safely under the 15 RPM free-tier cap


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class Frame:
    frame_id: str          # unique id assigned for this batch, e.g. "A", "B", ...
    episode: str
    path: str
    timestamp: float = 0.0 # optional: seconds into the fold, if known
    progress_raw: float = None    # raw VLM estimate, filled in later
    progress_smooth: float = None # isotonic-smoothed estimate, filled in later
    reasoning: str = ""


@dataclass
class FoldAnnotationJob:
    fold_name: str
    fold_description: str
    anchor_start_paths: List[str]
    anchor_end_paths: List[str]
    frames: List[Frame] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def encode_image(path: str):
    """Read an image file and return a Gemini-compatible inline image Part."""
    ext = Path(path).suffix.lower()
    mime_type = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
    }.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        data = f.read()
    return genai_types.Part.from_bytes(data=data, mime_type=mime_type)


def load_episode_frames(fold_dir: str, sample_per_episode: int = 10) -> List[Frame]:
    """
    Walk fold_dir/episode_*/frame_*.jpg, uniformly sample `sample_per_episode` frames
    per episode, and assign batch-local frame_ids (A, B, C...).
    """
    fold_path = Path(fold_dir)
    episodes = sorted([d for d in fold_path.iterdir() if d.is_dir()])
    all_frames = []
    letter_idx = 0

    def next_id(i):
        # A, B, ..., Z, AA, AB, ...
        letters = ""
        i += 1
        while i > 0:
            i, rem = divmod(i - 1, 26)
            letters = chr(65 + rem) + letters
        return letters

    for ep_dir in episodes:
        frame_files = sorted(ep_dir.glob("frame_*.jpg")) or sorted(ep_dir.glob("frame_*.png"))
        if not frame_files:
            continue
        # uniform sampling across the episode's fold clip
        idxs = np.linspace(0, len(frame_files) - 1, min(sample_per_episode, len(frame_files)))
        idxs = sorted(set(int(round(i)) for i in idxs))
        for i in idxs:
            fpath = frame_files[i]
            all_frames.append(Frame(
                frame_id=next_id(letter_idx),
                episode=ep_dir.name,
                path=str(fpath),
                timestamp=i / max(len(frame_files) - 1, 1),  # normalized 0..1 position in clip
            ))
            letter_idx += 1
    return all_frames


def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


# --------------------------------------------------------------------------- #
# VLM call
# --------------------------------------------------------------------------- #

def _anchor_episode_label(path: str) -> str:
    """Best-effort episode label for an anchor path, e.g. '.../anchors/episode_003/b1.jpg'
    -> 'episode_003' -- purely for the prompt text, falls back to the filename."""
    parts = Path(path).parts
    for p in reversed(parts[:-1]):
        if p.startswith("episode"):
            return p
    return Path(path).stem


def build_prompt_content(job: FoldAnnotationJob, batch: List[Frame]) -> List[Any]:
    """Construct the multimodal request content: anchors + shuffled frame batch + instructions.
    Returns a flat list of alternating text strings and genai Part image objects, which the
    Gemini SDK accepts directly as the `contents` argument."""
    content = []

    content.append(
        f"Fold stage: {job.fold_name}\n"
        f"Fold description: {job.fold_description}\n\n"
        f"ANCHOR_START examples (0% progress, canonical pre-fold state) -- "
        f"{len(job.anchor_start_paths)} example(s) from different episodes, all the SAME "
        f"0% concept, not a sequence:"
    )
    for path in job.anchor_start_paths:
        content.append(f"  ({_anchor_episode_label(path)}):")
        content.append(encode_image(path))

    content.append(
        f"ANCHOR_END examples (100% progress, canonical completed-fold state) -- "
        f"{len(job.anchor_end_paths)} example(s) from different episodes, all the SAME "
        f"100% concept, not a sequence:"
    )
    for path in job.anchor_end_paths:
        content.append(f"  ({_anchor_episode_label(path)}):")
        content.append(encode_image(path))

    content.append(
        f"\nNow here are {len(batch)} frames from different episodes of this same fold, "
        f"shuffled. Frames from different episodes are independent — do not assume "
        f"temporal order across episodes.\n"
    )

    for fr in batch:
        content.append(f"Frame {fr.frame_id} (episode: {fr.episode}):")
        content.append(encode_image(fr.path))

    content.append(
        "\nEstimate progress (0-100) for each frame based on visible physical fold state "
        "(crease depth, edge-to-edge alignment, paper curvature) — NOT hand position or "
        "assumed pacing. Progress may be nonlinear within the fold.\n\n"
        "Return ONLY valid JSON, no preamble, no markdown fences:\n"
        "{\n"
        '  "estimates": [\n'
        '    {"frame_id": "A", "episode": "episode_001", "progress": 0, "reasoning": "short phrase"},\n'
        "    ...\n"
        "  ]\n"
        "}"
    )

    return content


class DailyQuotaExhausted(RuntimeError):
    """Gemini returned a per-DAY (not per-minute) quota exhaustion. Retrying within
    this process cannot help -- the observed quotaId is
    'GenerateRequestsPerDayPerProjectPerModel-FreeTier', which only resets on Google's
    daily schedule, not within a MAX_RETRIES backoff window. Raised instead of
    retried so one exhausted batch fails the whole run immediately (propagating
    through run_gemini_progress.sh's `set -e`) instead of every remaining batch/fold
    separately burning ~35s of guaranteed-futile backoff before giving up."""


def call_vlm(client, job: FoldAnnotationJob, batch: List[Frame]) -> Dict[str, Any]:
    system_prompt = (
        "You are an expert annotator estimating fold-completion progress in origami "
        "paper-airplane folding videos. Judge each frame purely on visible physical state "
        "relative to the given anchor frames. Be consistent: the same physical state should "
        "get the same score regardless of which episode it came from."
    )
    content = build_prompt_content(job, batch)

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=content,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    # 2000 was too tight for a full BATCH_SIZE=12 response (each entry has
                    # frame_id/episode/progress/reasoning text) -- real 429-run logs showed
                    # repeated "Expecting property name"/"Unterminated string" JSON errors,
                    # the exact signature of a response getting cut off mid-generation.
                    # Retrying a truncated request with the same cap just truncates again,
                    # burning quota on doomed retries for a deterministic (not transient)
                    # cause. 8000 gives real headroom; it's a ceiling, not a cost.
                    max_output_tokens=8000,
                    response_mime_type="application/json",  # asks Gemini to return raw JSON
                ),
            )
            text = resp.text or ""
            clean = text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean)
        except (json.JSONDecodeError, Exception) as e:
            if "PerDay" in str(e):
                raise DailyQuotaExhausted(
                    f"Gemini daily quota exhausted for model={MODEL} -- retrying won't "
                    f"help until it resets (check https://ai.dev/rate-limit for exact "
                    f"timing, or try a different model -- quota is tracked per-model). "
                    f"Original error: {e}"
                ) from e
            wait = 2 ** attempt * 5  # backoff for transient/per-minute 429s: 5s, 10s, 20s
            if attempt == MAX_RETRIES - 1:
                print(f"  [WARN] batch failed after {MAX_RETRIES} attempts: {e}")
                return {"estimates": []}
            print(f"  [retry {attempt + 1}] {e} — waiting {wait}s")
            time.sleep(wait)
            continue


def annotate_fold(job: FoldAnnotationJob, api_key: str = None, max_batches: int = None) -> None:
    """Run VLM annotation over all frames in job.frames, in shuffled batches, in place.

    max_batches: if set, only process this many batches (for a cheap dry run before
    burning through your daily quota on the full frame set).
    """
    if genai is None:
        raise RuntimeError("pip install google-genai --break-system-packages")
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "No Gemini API key found. Get a free key at https://aistudio.google.com/apikey "
            "and either pass --api_key or set GEMINI_API_KEY."
        )
    client = genai.Client(api_key=key)

    frames_by_id = {f.frame_id: f for f in job.frames}
    shuffled = job.frames[:]
    random.shuffle(shuffled)

    batches = list(chunk(shuffled, BATCH_SIZE))
    if max_batches is not None:
        batches = batches[:max_batches]
        print(f"  [dry run] limiting to {len(batches)} batch(es) "
              f"({len(batches) * BATCH_SIZE} frames max, out of {len(shuffled)} loaded)")

    for i, batch in enumerate(batches):
        print(f"  Annotating batch {i + 1}/{len(batches)} ({len(batch)} frames)...")
        result = call_vlm(client, job, batch)
        for est in result.get("estimates", []):
            fr = frames_by_id.get(est.get("frame_id"))
            if fr is not None:
                fr.progress_raw = est.get("progress")
                fr.reasoning = est.get("reasoning", "")
        if i < len(batches) - 1:
            time.sleep(SECONDS_BETWEEN_CALLS)  # stay under free-tier RPM cap


# --------------------------------------------------------------------------- #
# Smoothing (isotonic regression per episode)
# --------------------------------------------------------------------------- #

def smooth_progress_per_episode(job: FoldAnnotationJob) -> None:
    """
    For each episode, fit isotonic regression of progress_raw vs. timestamp (normalized
    position in the fold clip). Enforces monotonic non-decreasing progress while preserving
    nonlinearity, and denoises individual VLM estimation errors.
    """
    by_episode: Dict[str, List[Frame]] = {}
    for fr in job.frames:
        by_episode.setdefault(fr.episode, []).append(fr)

    for ep, frames in by_episode.items():
        frames_valid = [f for f in frames if f.progress_raw is not None]
        if len(frames_valid) < 2:
            for f in frames_valid:
                f.progress_smooth = f.progress_raw
            continue
        frames_valid.sort(key=lambda f: f.timestamp)
        x = np.array([f.timestamp for f in frames_valid])
        y = np.array([f.progress_raw for f in frames_valid], dtype=float)

        iso = IsotonicRegression(y_min=0, y_max=100, increasing=True, out_of_bounds="clip")
        y_smooth = iso.fit_transform(x, y)

        for f, ys in zip(frames_valid, y_smooth):
            f.progress_smooth = float(ys)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #

def consistency_report(job: FoldAnnotationJob) -> Dict[str, Any]:
    """
    Basic cross-episode consistency checks:
      - monotonicity violation rate per episode (raw vs smoothed)
      - spread of progress estimates at matched normalized timestamps across episodes
    """
    by_episode: Dict[str, List[Frame]] = {}
    for fr in job.frames:
        if fr.progress_raw is not None:
            by_episode.setdefault(fr.episode, []).append(fr)

    report = {"per_episode": {}, "n_episodes": len(by_episode)}

    for ep, frames in by_episode.items():
        frames.sort(key=lambda f: f.timestamp)
        raw = [f.progress_raw for f in frames]
        violations = sum(1 for a, b in zip(raw, raw[1:]) if b < a)
        report["per_episode"][ep] = {
            "n_frames": len(frames),
            "monotonicity_violations": violations,
            "violation_rate": round(violations / max(len(raw) - 1, 1), 3),
        }

    # cross-episode spread: bucket by timestamp decile, compute std of smoothed progress
    buckets: Dict[int, List[float]] = {}
    for fr in job.frames:
        if fr.progress_smooth is None:
            continue
        decile = int(fr.timestamp * 10)
        buckets.setdefault(decile, []).append(fr.progress_smooth)

    spread = {
        f"decile_{d}": {"mean": round(float(np.mean(v)), 1), "std": round(float(np.std(v)), 1), "n": len(v)}
        for d, v in sorted(buckets.items())
    }
    report["cross_episode_spread_by_normalized_time"] = spread

    return report


# --------------------------------------------------------------------------- #
# Save / load
# --------------------------------------------------------------------------- #

def save_results(job: FoldAnnotationJob, out_path: str) -> None:
    rows = []
    for f in job.frames:
        rows.append({
            "episode": f.episode,
            "frame_path": f.path,
            "timestamp_norm": f.timestamp,
            "progress_raw": f.progress_raw,
            "progress_smooth": f.progress_smooth,
            "reasoning": f.reasoning,
        })
    with open(out_path, "w") as fp:
        json.dump({
            "fold_name": job.fold_name,
            "fold_description": job.fold_description,
            "anchor_start_paths": job.anchor_start_paths,
            "anchor_end_paths": job.anchor_end_paths,
            "frames": rows,
        }, fp, indent=2)
    print(f"Saved {len(rows)} frame annotations to {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_dir", required=True, help="Dir with episode_*/frame_*.jpg")
    parser.add_argument("--fold_name", required=True)
    parser.add_argument("--fold_description", default="")
    parser.add_argument("--anchor_start", required=True, nargs="+",
                         help="one or more 0%%-progress reference images (shell-glob friendly, "
                              "e.g. auto_progress/anchors/episode_*/b0.jpg) -- pool one per "
                              "episode rather than anchoring against a single episode")
    parser.add_argument("--anchor_end", required=True, nargs="+",
                         help="one or more 100%%-progress reference images, same as --anchor_start")
    parser.add_argument("--sample_per_episode", type=int, default=10)
    parser.add_argument("--out", default="fold_progress_results.json")
    parser.add_argument("--api_key", default=None, help="Defaults to GEMINI_API_KEY env var")
    parser.add_argument("--max_batches", type=int, default=None,
                         help="Limit to N batches for a cheap dry run before processing everything")
    args = parser.parse_args()

    job = FoldAnnotationJob(
        fold_name=args.fold_name,
        fold_description=args.fold_description,
        anchor_start_paths=args.anchor_start,
        anchor_end_paths=args.anchor_end,
    )
    job.frames = load_episode_frames(args.fold_dir, args.sample_per_episode)
    print(f"Loaded {len(job.frames)} frames across "
          f"{len(set(f.episode for f in job.frames))} episodes.")
    print(f"Anchors: {len(job.anchor_start_paths)} start / {len(job.anchor_end_paths)} end "
          f"({[_anchor_episode_label(p) for p in job.anchor_start_paths]})")

    annotate_fold(job, api_key=args.api_key, max_batches=args.max_batches)
    smooth_progress_per_episode(job)
    save_results(job, args.out)

    report = consistency_report(job)
    print("\n--- Consistency report ---")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

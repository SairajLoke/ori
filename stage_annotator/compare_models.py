#!/usr/bin/env python3
"""Compare candidate Gemini models/thinking-levels across ALL 30 frames the real
run already scored (not just 2 spot-checked ones), so the comparison has real
signal instead of anecdote. Reuses fold_progress_pipeline.py's own sampling
(load_episode_frames) so it's the exact same 30 frames as fold_progress_results.json,
batched the same way (BATCH_SIZE=12) -- an apples-to-apples rerun per model.

Since there's no ground truth, the primary comparison metric is each model's OWN
internal consistency: monotonicity-violation rate per episode (same metric
fold_progress_pipeline.py's consistency_report already uses) -- a model that
contradicts itself less across a fold's progression is more trustworthy, ground
truth or not. Also reports mean |diff| vs. the already-recorded baseline scores.

Usage:
    export GEMINI_API_KEY=your_key
    python compare_models.py
"""
import json
import os
import pathlib
import random
import sys
import time

from google import genai
from google.genai import types as genai_types

from fold_progress_pipeline import load_episode_frames, chunk, encode_image

FOLD_DIR = "auto_progress/fold0_frames"
ANCHOR_START = "auto_progress/anchors/episode_000/b0.jpg"
ANCHOR_END = "auto_progress/anchors/episode_000/b1.jpg"
FOLD_NAME = "triangle_fold_1"
FOLD_DESC = "bringing left wing edge to align with center crease"
BASELINE_RESULTS = "fold_progress_results.json"
BATCH_SIZE = 12
SECONDS_BETWEEN_CALLS = 4.5

CANDIDATES = [
    ("gemini-3.6-flash (current default)", "gemini-3.6-flash", None),
    ("gemini-3.6-flash (thinking=HIGH)", "gemini-3.6-flash", "high"),
    ("gemini-3.7-flash", "gemini-3.7-flash", None),
    ("gemini-robotics-er-2-preview", "gemini-robotics-er-2-preview", None),
]


def build_batch_content(batch):
    content = [
        f"Fold stage: {FOLD_NAME}\nFold description: {FOLD_DESC}\n\n"
        f"ANCHOR_START (0% progress, canonical pre-fold state):",
        encode_image(ANCHOR_START),
        "ANCHOR_END (100% progress, canonical completed-fold state):",
        encode_image(ANCHOR_END),
        f"\nNow here are {len(batch)} frames from different episodes, shuffled. "
        f"Frames from different episodes are independent -- do not assume temporal "
        f"order across episodes.\n",
    ]
    for fr in batch:
        content.append(f"Frame {fr.frame_id} (episode: {fr.episode}):")
        content.append(encode_image(fr.path))
    content.append(
        "\nEstimate progress (0-100) for each frame based on visible physical fold state "
        "(crease depth, edge-to-edge alignment, paper curvature) -- NOT hand position or "
        "assumed pacing.\n\n"
        "Return ONLY valid JSON, no preamble, no markdown fences:\n"
        '{"estimates": [{"frame_id": "A", "episode": "episode_001", "progress": 0, '
        '"reasoning": "short phrase"}, ...]}'
    )
    return content


def score_all(client, model_id, thinking_level, frames):
    system_prompt = (
        "You are an expert annotator estimating fold-completion progress in origami "
        "paper-airplane folding videos. Judge each frame purely on visible physical state "
        "relative to the given anchor frames. Be consistent: the same physical state should "
        "get the same score regardless of which episode it came from."
    )
    shuffled = frames[:]
    random.shuffle(shuffled)
    batches = list(chunk(shuffled, BATCH_SIZE))
    results = {}  # frame_id -> (progress, reasoning)
    for i, batch in enumerate(batches):
        config_kwargs = dict(
            system_instruction=system_prompt, max_output_tokens=3000,
            response_mime_type="application/json",
        )
        if thinking_level is not None:
            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(thinking_level=thinking_level)
        try:
            resp = client.models.generate_content(
                model=model_id, contents=build_batch_content(batch),
                config=genai_types.GenerateContentConfig(**config_kwargs),
            )
            text = (resp.text or "").replace("```json", "").replace("```", "").strip()
            # Some models (seen with robotics-er-2-preview) append content after the
            # JSON object despite response_mime_type=application/json. raw_decode parses
            # just the first valid JSON value and reports where it ended, instead of
            # json.loads's all-or-nothing "Extra data" failure on the whole response.
            parsed, _end = json.JSONDecoder().raw_decode(text)
            for est in parsed.get("estimates", []):
                results[est.get("frame_id")] = (est.get("progress"), est.get("reasoning", ""))
        except Exception as e:
            print(f"    batch {i+1}/{len(batches)} FAILED: {e}")
        if i < len(batches) - 1:
            time.sleep(SECONDS_BETWEEN_CALLS)
    return results


def monotonicity_violations(frames, results):
    """Same metric as fold_progress_pipeline.py's consistency_report, computed per
    episode then summed, on THIS model's raw scores."""
    by_ep = {}
    for fr in frames:
        val = results.get(fr.frame_id, (None, None))[0]
        if val is not None:
            by_ep.setdefault(fr.episode, []).append((fr.timestamp, val))
    total_violations, total_pairs = 0, 0
    for ep, pts in by_ep.items():
        pts.sort(key=lambda x: x[0])
        vals = [v for _, v in pts]
        total_violations += sum(1 for a, b in zip(vals, vals[1:]) if b < a)
        total_pairs += max(len(vals) - 1, 0)
    return total_violations, total_pairs


def main():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("export GEMINI_API_KEY first")
    client = genai.Client(api_key=key)

    frames = load_episode_frames(FOLD_DIR, sample_per_episode=10)
    print(f"Loaded {len(frames)} frames across {len(set(f.episode for f in frames))} episodes "
          f"(same set fold_progress_results.json already scored once).\n")

    baseline = {}
    if os.path.exists(BASELINE_RESULTS):
        d = json.load(open(BASELINE_RESULTS))
        for row in d["frames"]:
            # frame_path in the baseline JSON is absolute (from the original run's
            # --fold_dir); FOLD_DIR here is relative -- resolve both so lookups actually
            # match instead of silently missing every time (that's why mean_diff was nan).
            baseline[str(pathlib.Path(row["frame_path"]).resolve())] = row["progress_raw"]

    summary = []
    for label, model_id, thinking_level in CANDIDATES:
        print(f"=== {label} ===")
        results = score_all(client, model_id, thinking_level, frames)
        n_scored = sum(1 for v, _ in results.values() if v is not None)
        violations, pairs = monotonicity_violations(frames, results)
        diffs = []
        for fr in frames:
            val = results.get(fr.frame_id, (None, None))[0]
            base = baseline.get(str(pathlib.Path(fr.path).resolve()))
            if val is not None and base is not None:
                diffs.append(abs(val - base))
        mean_diff = sum(diffs) / len(diffs) if diffs else float("nan")
        rate = violations / pairs if pairs else float("nan")
        print(f"  scored {n_scored}/{len(frames)} frames, "
              f"monotonicity violations {violations}/{pairs} ({rate:.1%}), "
              f"mean |diff vs baseline 3.6-flash run| = {mean_diff:.1f}\n")
        summary.append((label, n_scored, violations, pairs, rate, mean_diff))

    print("\n" + "=" * 70)
    print(f"{'model':<38} {'scored':>7} {'violations':>12} {'viol_rate':>10} {'mean_diff':>10}")
    for label, n_scored, violations, pairs, rate, mean_diff in summary:
        print(f"{label:<38} {n_scored:>7} {violations:>5}/{pairs:<6} {rate:>9.1%} {mean_diff:>10.1f}")


if __name__ == "__main__":
    main()

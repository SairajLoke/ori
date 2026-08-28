"""
nemotron_stage_progress.py

Like fold_progress_pipeline.py, but (a) uses NVIDIA's hosted Nemotron-3-Nano-Omni
VLM instead of Gemini, and (b) instead of a shuffled cross-episode batch judged
against explicit 0%/100% anchor images, it sends only the first OR last
`n_frames` frames of each episode's stage clip (--position start/end) in one
call per episode, and asks the model to judge the fold state at that boundary.
Useful as a lightweight sanity check on annotate.py's marked stage boundaries
-- did the stage start clean (--position start) and finish cleanly
(--position end)? -- without paying for a full uniformly-sampled,
anchor-calibrated pass.

Why Nemotron over Gemini here (verified against https://build.nvidia.com/nvidia/
nemotron-3-nano-omni-30b-a3b-reasoning on 2026-08-27):
  - Rate limits on a build.nvidia.com API key: 40 requests/minute, 10,000
    requests/day. Contrast with fold_progress_pipeline.py's Gemini setup,
    where the actual observed free-tier cap for gemini-3.6-flash is 20
    requests/day (see that file's docstring) -- Nemotron's daily cap is
    ~500x larger, so per-episode calls (instead of Gemini's frame-batching
    to conserve quota) are comfortably affordable.
  - Natively multimodal: image_url/video_url/audio_url content parts,
    262K token context window -- a handful of images in one call is trivial.
  - It is an OpenAI-Chat-Completions-compatible REST endpoint
    (https://integrate.api.nvidia.com/v1/chat/completions), so this uses
    plain `requests` (already a project dependency) instead of a new SDK.
  - `structuredOutput` is reported False for this model (no `response_format`
    field in its request schema) -- like the Gemini path, we ask for JSON in
    the prompt and parse leniently (markdown-fence stripping), rather than
    relying on a JSON-mode guarantee that doesn't exist here.

Usage:
    export NVIDIA_API_KEY=nvapi-...   # get one free at https://build.nvidia.com
    python nemotron_stage_progress.py --fold_dir auto_progress/fold3_frames \
        --fold_name "wing valley fold" \
        --fold_description "folding the wing crease outward" \
        --position end --n_frames 5

    Same, but check the START of the stage instead:
    python nemotron_stage_progress.py --fold_dir auto_progress/fold3_frames \
        --fold_name "wing valley fold" --position start --n_frames 5

    Dry run first (only processes 1 episode):
    python nemotron_stage_progress.py --fold_dir auto_progress/fold3_frames \
        --fold_name "wing valley fold" --max_episodes 1

Expected frame directory layout (same as fold_progress_pipeline.py / produced
by annotate.py --export_frames_root):
    fold_dir/
        episode_000/
            frame_0000.jpg   <- --position start takes frames from this end
            ...
            frame_0138.jpg   <- --position end takes frames from this end
        episode_001/
            ...

Requires: pip install requests   (already present in this project's .venv)
"""

import os
import json
import time
import base64
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import requests


INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
MAX_RETRIES = 3
SECONDS_BETWEEN_CALLS = 1.6   # ~37 rpm, safely under the 40 rpm cap

DEFAULT_QUESTION_END = (
    "Based ONLY on these final frames of the stage (shown in chronological "
    "order, ending at the last frame recorded for this stage), estimate how "
    "visually complete this fold looks at the END of the clip, and whether "
    "the fold looks cleanly finished (not mid-motion, not abandoned "
    "half-way) by the last frame."
)

DEFAULT_QUESTION_START = (
    "Based ONLY on these first frames of the stage (shown in chronological "
    "order, starting at the first frame recorded for this stage), estimate "
    "how far along the fold already looks at the START of the clip, and "
    "whether this is a clean starting point for the stage -- i.e. the paper "
    "is still in the pre-fold state this stage is supposed to start from, "
    "not already showing leftover motion or partial folding carried over "
    "from the previous stage."
)

DEFAULT_QUESTION_GROUP = (
    "The two ANCHOR images above define your 0-100 scale for THIS episode: "
    "ANCHOR_START = 0 (the state before this stage's fold begins), "
    "ANCHOR_END = 100 (this stage's fold fully, cleanly complete).\n\n"
    "The {n} GROUP frames below are consecutive frames sampled from "
    "somewhere in the middle of this same episode's clip -- they are NOT "
    "necessarily near the start or end just because of where they appear in "
    "this message. First, briefly identify the physical action/motion "
    "visible across these {n} frames. Then, using the anchors to calibrate "
    "your scale, rate fold-completion progress (0-100) for EACH of the {n} "
    "frames individually, in the same order they were shown -- do not just "
    "rate the last one."
)

_MIME_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".webp": "image/webp",
}


@dataclass
class StageEndJob:
    fold_name: str
    fold_description: str
    fold_dir: str
    n_frames: int = 5
    position: str = "end"   # "end" (last n frames) or "start" (first n frames)
    question: str = None    # None -> resolved from `position` in __post_init__
    results: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.position not in ("start", "end"):
            raise ValueError(f"position must be 'start' or 'end', got {self.position!r}")
        if self.question is None:
            self.question = DEFAULT_QUESTION_START if self.position == "start" else DEFAULT_QUESTION_END


@dataclass
class ProgressCurveJob:
    fold_name: str
    fold_description: str
    fold_dir: str
    group_size: int = 5      # consecutive frames scored together per call
    question: str = DEFAULT_QUESTION_GROUP
    results: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)  # episode -> per-frame results


class DailyQuotaExhausted(RuntimeError):
    """A 429 whose body indicates a per-day (not per-minute) cap. At the
    verified 10,000 requests/day limit for this model this should be very
    hard to hit from this script's own usage, but is handled the same way
    fold_progress_pipeline.py handles Gemini's per-day cap: fail the whole
    run immediately rather than burning MAX_RETRIES of guaranteed-futile
    backoff on every remaining episode."""


# --------------------------------------------------------------------------- #
# Frame loading
# --------------------------------------------------------------------------- #

def encode_image_data_url(path: str) -> str:
    """Read an image file and return an OpenAI/NIM-compatible inline
    `data:<mime>;base64,<...>` URL, for use as an `image_url.url` value."""
    mime = _MIME_TYPES.get(Path(path).suffix.lower(), "image/jpeg")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def load_boundary_frames(fold_dir: str, n: int, position: str = "end") -> Dict[str, List[str]]:
    """fold_dir/episode_*/frame_*.{jpg,png} -> {episode_name: [n frame paths,
    oldest to newest]}, taking the first n frames if position=="start" or the
    last n if position=="end". Episodes with zero frames are skipped."""
    fold_path = Path(fold_dir)
    episodes = sorted(d for d in fold_path.iterdir() if d.is_dir())
    out = {}
    for ep_dir in episodes:
        frame_files = sorted(ep_dir.glob("frame_*.jpg")) or sorted(ep_dir.glob("frame_*.png"))
        if not frame_files:
            continue
        out[ep_dir.name] = [str(p) for p in (frame_files[:n] if position == "start" else frame_files[-n:])]
    return out


def load_episode_frame_lists(fold_dir: str) -> Dict[str, List[str]]:
    """fold_dir/episode_*/frame_*.{jpg,png} -> {episode_name: [ALL frame
    paths, oldest to newest]}. Episodes with zero frames are skipped."""
    fold_path = Path(fold_dir)
    episodes = sorted(d for d in fold_path.iterdir() if d.is_dir())
    out = {}
    for ep_dir in episodes:
        frame_files = sorted(ep_dir.glob("frame_*.jpg")) or sorted(ep_dir.glob("frame_*.png"))
        if frame_files:
            out[ep_dir.name] = [str(p) for p in frame_files]
    return out


def chunk_from_end(frames: List[str], size: int) -> List[List[str]]:
    """Partition `frames` into consecutive, NON-overlapping groups of exactly
    `size`, anchored to the END of the list so the last group always ends
    exactly at the true last frame. Any remainder (n % size frames) is
    dropped from the START rather than padded -- per-episode edge frames
    that can't form a full group are simply not scored by this pass, rather
    than being scored from a padded/duplicated window (which was tried and
    produced unreliable scores -- see the design discussion this replaced)."""
    n = len(frames)
    n_groups = n // size
    start = n - n_groups * size
    return [frames[start + g * size: start + (g + 1) * size] for g in range(n_groups)]


# --------------------------------------------------------------------------- #
# VLM call
# --------------------------------------------------------------------------- #

def build_messages(job: StageEndJob, episode: str, frame_paths: List[str]) -> List[Dict[str, Any]]:
    system_prompt = (
        "You are an expert annotator judging fold-completion state in origami "
        "paper-airplane folding videos. Judge purely on visible physical state "
        "(crease depth, edge-to-edge alignment, paper curvature) -- not on hand "
        "position or assumed pacing. Respond with ONLY a single valid JSON "
        "object, no markdown fences, no preamble."
    )
    boundary_word = "first" if job.position == "start" else "last"
    anchor_word = "first frame" if job.position == "start" else "final image"
    content: List[Dict[str, Any]] = [{
        "type": "text",
        "text": (
            f"Fold stage: {job.fold_name}\n"
            f"Fold description: {job.fold_description}\n"
            f"Episode: {episode}\n\n"
            f"Here are the {boundary_word} {len(frame_paths)} recorded frames of "
            f"this episode's stage clip, in chronological order (the "
            f"{anchor_word} is the {job.position} of this stage):"
        ),
    }]
    for i, p in enumerate(frame_paths):
        offset = i if job.position == "start" else len(frame_paths) - 1 - i
        sign = "+" if job.position == "start" else "-"
        content.append({"type": "text", "text": f"Frame t{sign}{offset}:"})
        content.append({"type": "image_url", "image_url": {"url": encode_image_data_url(p)}})

    quality_field = ('"clean_start": <true|false>  // true if this is a clean pre-fold '
                      'starting point, not bleeding over from the previous stage'
                      if job.position == "start" else
                      '"stage_complete": <true|false>  // true if the fold looks cleanly '
                      'finished by the last frame')
    progress_field = "progress_at_start" if job.position == "start" else "progress_at_end"
    content.append({
        "type": "text",
        "text": (
            f"{job.question}\n\n"
            "Return ONLY this JSON object (no comments in your actual output):\n"
            "{\n"
            f'  "episode": "{episode}",\n'
            f'  "{progress_field}": <0-100 integer>,\n'
            f'  {quality_field},\n'
            '  "reasoning": "<short phrase>"\n'
            "}"
        ),
    })

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def build_group_messages(job: ProgressCurveJob, episode: str, anchor_start: str, anchor_end: str,
                          group_paths: List[str]) -> List[Dict[str, Any]]:
    """ONE call scores ALL frames in group_paths together (a consecutive,
    non-overlapping chunk from chunk_from_end), calibrated against this
    episode's OWN first/last frame sent as explicit anchor images -- fixes
    the earlier design, where a frame-by-frame call with only a text
    description of the stage (no visual anchor) let the model's 0/100 scale
    drift: the same frame_0000 scored 0 when explicitly framed as "the
    stage's start" but 95 when shown with no positional framing at all,
    because the model had no visual reference for what 0%/100% actually
    look like for THIS episode -- only a text description to go on."""
    system_prompt = (
        "You are an expert annotator estimating fold-completion progress in "
        "origami paper-airplane folding videos. Judge purely on visible "
        "physical state (crease depth, edge-to-edge alignment, paper "
        "curvature) -- not hand position or assumed pacing. Respond with "
        "ONLY a single valid JSON object, no markdown fences, no preamble."
    )
    n = len(group_paths)
    content: List[Dict[str, Any]] = [{
        "type": "text",
        "text": (
            f"Fold stage: {job.fold_name}\n"
            f"Fold description: {job.fold_description}\n"
            f"Episode: {episode}\n"
        ),
    }, {
        "type": "text",
        "text": "ANCHOR_START (0% -- state before this stage's fold begins):",
    }, {
        "type": "image_url", "image_url": {"url": encode_image_data_url(anchor_start)},
    }, {
        "type": "text",
        "text": "ANCHOR_END (100% -- this stage's fold fully, cleanly complete):",
    }, {
        "type": "image_url", "image_url": {"url": encode_image_data_url(anchor_end)},
    }, {
        "type": "text",
        "text": f"GROUP: {n} consecutive frames from somewhere in the middle of this episode:",
    }]
    for i, p in enumerate(group_paths):
        content.append({"type": "text", "text": f"Group frame {i + 1} of {n}:"})
        content.append({"type": "image_url", "image_url": {"url": encode_image_data_url(p)}})

    content.append({
        "type": "text",
        "text": (
            f"{job.question.format(n=n)}\n\n"
            "Return ONLY this JSON object:\n"
            "{\n"
            '  "action_observed": "<short phrase>",\n'
            f'  "progress_scores": [<{n} integers 0-100, one per group frame, in order>]\n'
            "}"
        ),
    })

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def call_nemotron(messages: List[Dict[str, Any]], api_key: str, model: str = MODEL,
                   reasoning: bool = False, reasoning_budget: int = 1024) -> Dict[str, Any]:
    """POST one chat-completion request. `reasoning=False` (default) sets
    chat_template_kwargs.enable_thinking=False for a fast, direct JSON
    response -- this mirrors NVIDIA's own documented image-example default
    (see build.nvidia.com's Python snippet for this model). Pass
    reasoning=True for the model's own chain-of-thought before answering
    (slower, costs more output tokens against the 65536 max_tokens ceiling,
    budget controlled by reasoning_budget)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.2,   # judgment task, not creative generation -- keep it deterministic-ish
        "top_p": 0.95,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": reasoning},
    }
    if reasoning:
        payload["reasoning_budget"] = reasoning_budget

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(INVOKE_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                body = resp.text
                if "day" in body.lower() and "minute" not in body.lower():
                    raise DailyQuotaExhausted(
                        f"Nemotron daily quota exhausted (model={model}) -- retrying won't "
                        f"help until it resets. Original response: {body}"
                    )
                wait = 2 ** attempt * 5
                if attempt == MAX_RETRIES - 1:
                    return {"error": f"rate-limited after {MAX_RETRIES} attempts: {body}"}
                print(f"  [rate limited, retry {attempt + 1}] waiting {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"] or ""
            clean = text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean)
        except DailyQuotaExhausted:
            raise
        except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
            wait = 2 ** attempt * 5
            if attempt == MAX_RETRIES - 1:
                print(f"  [WARN] episode call failed after {MAX_RETRIES} attempts: {e}")
                return {"error": str(e)}
            print(f"  [retry {attempt + 1}] {e} -- waiting {wait}s")
            time.sleep(wait)
    return {"error": "unreachable"}


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def annotate_stage_boundary(job: StageEndJob, api_key: str, model: str = MODEL,
                             reasoning: bool = False, max_episodes: Optional[int] = None,
                             seconds_between_calls: float = SECONDS_BETWEEN_CALLS) -> None:
    frames_by_episode = load_boundary_frames(job.fold_dir, job.n_frames, job.position)
    episodes = sorted(frames_by_episode)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
        print(f"  [dry run] limiting to {len(episodes)} episode(s)")

    boundary_word = "first" if job.position == "start" else "last"
    for i, ep in enumerate(episodes):
        frame_paths = frames_by_episode[ep]
        print(f"  [{i + 1}/{len(episodes)}] {ep}: sending {boundary_word} {len(frame_paths)} frames...")
        messages = build_messages(job, ep, frame_paths)
        result = call_nemotron(messages, api_key, model=model, reasoning=reasoning)
        result.setdefault("episode", ep)
        result["frame_paths"] = frame_paths
        job.results.append(result)
        if i < len(episodes) - 1:
            time.sleep(seconds_between_calls)


def save_results(job: StageEndJob, out_path: str) -> None:
    with open(out_path, "w") as fp:
        json.dump({
            "fold_name": job.fold_name,
            "fold_description": job.fold_description,
            "fold_dir": job.fold_dir,
            "n_frames": job.n_frames,
            "position": job.position,
            "question": job.question,
            "results": job.results,
        }, fp, indent=2)
    print(f"Saved {len(job.results)} episode judgments to {out_path}")


def annotate_progress_groups(job: ProgressCurveJob, api_key: str, model: str = MODEL,
                              reasoning: bool = False, max_episodes: Optional[int] = None,
                              max_groups_per_episode: Optional[int] = None,
                              seconds_between_calls: float = SECONDS_BETWEEN_CALLS,
                              checkpoint_path: Optional[str] = None) -> None:
    """One call per GROUP of job.group_size consecutive frames -- every frame
    in the group gets its own score from that single call (see
    build_group_messages), calibrated against this episode's own first/last
    frame sent as anchor images. ~group_size x fewer calls than one-call-
    per-frame, with (unlike the sliding-window design this replaced) a
    consistent 0/100 reference the model can actually see, not just a text
    description. Frames left over at the very start of an episode (n %
    group_size of them) are not scored by this pass -- see chunk_from_end.

    If `checkpoint_path` is set, the full (partial) result set is written to
    it after EVERY group call, not just at the end -- this is a multi-minute,
    multi-call run against a real network endpoint that has shown transient
    503s and a DailyQuotaExhausted path exists that would otherwise propagate
    straight out of this function with nothing saved at all. Without this,
    a crash on group 90/110 would lose the other 89 already-paid-for calls."""
    frames_by_episode = load_episode_frame_lists(job.fold_dir)
    episodes = sorted(frames_by_episode)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
        print(f"  [dry run] limiting to {len(episodes)} episode(s)")

    total_calls = 0
    try:
        for ep in episodes:
            frames = frames_by_episode[ep]
            anchor_start, anchor_end = frames[0], frames[-1]
            groups = chunk_from_end(frames, job.group_size)
            n_groups = len(groups)
            start_offset = len(frames) - n_groups * job.group_size  # dropped leading frames
            if max_groups_per_episode is not None:
                groups = groups[:max_groups_per_episode]

            job.results.setdefault(ep, [])
            print(f"  {ep}: {len(groups)}/{n_groups} group(s) of {job.group_size} "
                  f"({start_offset} leading frame(s) not scored)...")
            for g, group in enumerate(groups):
                messages = build_group_messages(job, ep, anchor_start, anchor_end, group)
                result = call_nemotron(messages, api_key, model=model, reasoning=reasoning)
                scores = result.get("progress_scores")
                action = result.get("action_observed", result.get("error"))
                if not isinstance(scores, list) or len(scores) != len(group):
                    print(f"  [WARN] group {g} on {ep}: expected {len(group)} scores, got {scores!r}")
                    scores = [None] * len(group)
                base_idx = start_offset + g * job.group_size
                for i, (frame_path, score) in enumerate(zip(group, scores)):
                    job.results[ep].append({
                        "frame_index": base_idx + i,
                        "frame_path": frame_path,
                        "progress": score,
                        "action_observed": action,
                        "group": g,
                    })
                total_calls += 1
                if checkpoint_path:
                    save_curve_results(job, checkpoint_path, quiet=True)
                is_last_call = (ep == episodes[-1]) and (g == len(groups) - 1)
                if not is_last_call:
                    time.sleep(seconds_between_calls)
    finally:
        if checkpoint_path:
            save_curve_results(job, checkpoint_path, quiet=True)
    print(f"  done: {total_calls} total VLM calls across {len(episodes)} episode(s)")


def retry_missing_groups(existing_path: str, api_key: str, model: str = MODEL,
                          reasoning: bool = False,
                          seconds_between_calls: float = SECONDS_BETWEEN_CALLS) -> ProgressCurveJob:
    """Load an existing --mode curve results file, find every group that has
    at least one null `progress` (a call that failed after MAX_RETRIES --
    e.g. from the 503s NVIDIA's endpoint was returning during the fold3
    run), and re-run ONLY those specific groups, updating the same entries
    in place (matched by frame_index) rather than appending duplicates.
    Already-successful groups are left untouched and re-saved unchanged."""
    data = json.loads(Path(existing_path).read_text())
    job = ProgressCurveJob(
        fold_name=data["fold_name"], fold_description=data["fold_description"],
        fold_dir=data["fold_dir"], group_size=data["group_size"], question=data["question"],
    )
    job.results = data["results"]

    frames_by_episode = load_episode_frame_lists(job.fold_dir)

    missing_by_ep = {}
    for ep, rows in job.results.items():
        bad_groups = sorted({r["group"] for r in rows if r["progress"] is None})
        if bad_groups:
            missing_by_ep[ep] = bad_groups
    if not missing_by_ep:
        print("  no missing groups found -- nothing to retry")
        return job

    total_missing_groups = sum(len(v) for v in missing_by_ep.values())
    print(f"  retrying {total_missing_groups} missing group(s) across {len(missing_by_ep)} episode(s)")

    calls_done = 0
    for ep, bad_group_indices in missing_by_ep.items():
        frames = frames_by_episode[ep]
        anchor_start, anchor_end = frames[0], frames[-1]
        groups = chunk_from_end(frames, job.group_size)
        start_offset = len(frames) - len(groups) * job.group_size
        by_index = {r["frame_index"]: r for r in job.results[ep]}

        for g in bad_group_indices:
            group = groups[g]
            print(f"  {ep} group {g}: retrying...")
            messages = build_group_messages(job, ep, anchor_start, anchor_end, group)
            result = call_nemotron(messages, api_key, model=model, reasoning=reasoning)
            scores = result.get("progress_scores")
            action = result.get("action_observed", result.get("error"))
            if not isinstance(scores, list) or len(scores) != len(group):
                print(f"  [WARN] group {g} on {ep}: still failing -- expected {len(group)} scores, got {scores!r}")
                scores = [None] * len(group)
            base_idx = start_offset + g * job.group_size
            for i, score in enumerate(scores):
                entry = by_index[base_idx + i]
                entry["progress"] = score
                entry["action_observed"] = action
            calls_done += 1
            save_curve_results(job, existing_path, quiet=True)
            is_last = (ep == list(missing_by_ep)[-1]) and (g == bad_group_indices[-1])
            if not is_last:
                time.sleep(seconds_between_calls)

    still_missing = sum(1 for rows in job.results.values() for r in rows if r["progress"] is None)
    print(f"  done: {calls_done} retry call(s); {still_missing} frame(s) still null")
    return job


def save_curve_results(job: ProgressCurveJob, out_path: str, quiet: bool = False) -> None:
    # Write to a temp file + rename so a reader (e.g. you, tailing progress
    # mid-run) never sees a half-written file if this races a read.
    tmp_path = f"{out_path}.tmp"
    with open(tmp_path, "w") as fp:
        json.dump({
            "fold_name": job.fold_name,
            "fold_description": job.fold_description,
            "fold_dir": job.fold_dir,
            "group_size": job.group_size,
            "question": job.question,
            "results": job.results,
        }, fp, indent=2)
    os.replace(tmp_path, out_path)
    if not quiet:
        total = sum(len(v) for v in job.results.values())
        print(f"Saved {total} per-frame scores across {len(job.results)} episode(s) to {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["boundary", "curve"], default="boundary",
                         help="'boundary' (default) = one score per episode, from its first/last "
                              "n_frames (see --position). 'curve' = every frame gets its own "
                              "score, group_size consecutive frames at a time, calibrated "
                              "against that episode's own first/last frame as anchor images.")
    parser.add_argument("--fold_dir", default=None,
                         help="Dir with episode_*/frame_*.jpg -- required unless --retry_missing")
    parser.add_argument("--fold_name", default=None,
                         help="required unless --retry_missing")
    parser.add_argument("--fold_description", default="")
    parser.add_argument("--retry_missing", default=None, metavar="RESULTS_JSON",
                         help="[curve only] Instead of a full run, load an existing curve results "
                              "file, re-run only the groups that came back with null scores "
                              "(failed after MAX_RETRIES), and update it in place. Ignores "
                              "--fold_dir/--fold_name/--fold_description/--group_size/--question "
                              "-- those are read from the existing file.")
    # --mode boundary
    parser.add_argument("--n_frames", type=int, default=5,
                         help="[boundary] How many boundary frames to send per episode (4-5 typical)")
    parser.add_argument("--position", choices=["start", "end"], default="end",
                         help="[boundary] 'end' = last n frames of the stage (default), "
                              "'start' = first n frames of the stage")
    # --mode curve
    parser.add_argument("--group_size", type=int, default=5,
                         help="[curve] Consecutive frames scored together per call (5 typical)")
    parser.add_argument("--max_groups_per_episode", type=int, default=None,
                         help="[curve] Limit to N groups per episode for a cheap dry run")
    parser.add_argument("--question", default=None,
                         help="Override the judgment task given to the model "
                              "(defaults to a mode/position-appropriate prompt)")
    parser.add_argument("--out", default=None,
                         help="Defaults to nemotron_stage_{boundary,curve}_results.json")
    parser.add_argument("--api_key", default=None, help="Defaults to NVIDIA_API_KEY env var")
    parser.add_argument("--model", default=MODEL,
                         help="Override for a local NIM deployment (e.g. served under a different tag)")
    parser.add_argument("--reasoning", action="store_true",
                         help="Enable the model's chain-of-thought (chat_template_kwargs.enable_thinking) "
                              "before it answers -- slower, more output tokens, off by default")
    parser.add_argument("--max_episodes", type=int, default=None,
                         help="Limit to N episodes for a cheap dry run")
    parser.add_argument("--seconds_between_calls", type=float, default=SECONDS_BETWEEN_CALLS)
    args = parser.parse_args()

    key = args.api_key or os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise RuntimeError(
            "No NVIDIA API key found. Get a free key at https://build.nvidia.com "
            "and either pass --api_key or set NVIDIA_API_KEY."
        )

    if args.retry_missing:
        job = retry_missing_groups(args.retry_missing, api_key=key, model=args.model,
                                    reasoning=args.reasoning,
                                    seconds_between_calls=args.seconds_between_calls)
        save_curve_results(job, args.retry_missing)
        return

    if args.fold_dir is None or args.fold_name is None:
        raise RuntimeError("--fold_dir and --fold_name are required unless --retry_missing is given")

    if args.mode == "curve":
        out_path = args.out or "nemotron_stage_curve_results.json"
        job = ProgressCurveJob(
            fold_name=args.fold_name,
            fold_description=args.fold_description,
            fold_dir=args.fold_dir,
            group_size=args.group_size,
            question=args.question or DEFAULT_QUESTION_GROUP,
        )
        # out_path doubles as a live checkpoint: rewritten after every group
        # call (not just at the end), so a crash or Ctrl-C partway through a
        # long multi-episode run doesn't lose already-completed groups, and
        # you can inspect progress by reading out_path while it's still running.
        annotate_progress_groups(job, api_key=key, model=args.model, reasoning=args.reasoning,
                                  max_episodes=args.max_episodes,
                                  max_groups_per_episode=args.max_groups_per_episode,
                                  seconds_between_calls=args.seconds_between_calls,
                                  checkpoint_path=out_path)
        save_curve_results(job, out_path)
    else:
        job = StageEndJob(
            fold_name=args.fold_name,
            fold_description=args.fold_description,
            fold_dir=args.fold_dir,
            n_frames=args.n_frames,
            position=args.position,
            question=args.question,
        )
        annotate_stage_boundary(job, api_key=key, model=args.model, reasoning=args.reasoning,
                                 max_episodes=args.max_episodes,
                                 seconds_between_calls=args.seconds_between_calls)
        save_results(job, args.out or "nemotron_stage_boundary_results.json")


if __name__ == "__main__":
    main()

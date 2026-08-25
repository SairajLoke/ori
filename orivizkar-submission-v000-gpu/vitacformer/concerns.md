# Open concerns

Working notes on things that are wrong, risky, or unverified. Everything here was
measured against the real data (`season_POC22061`, 91338 frames) or read out of
the competition spec in
`ori/origami-inference-kit-participant/docs/` — none of it is inferred from
intuition alone. Resolved items are listed at the bottom for reference.

Status key: **OPEN** = not addressed · **DECIDE** = needs a call from you ·
**ASK** = needs the organizers · **UNTESTED** = built but never exercised

---

## 1. Image size does not match the deployment contract — **OPEN**

`robot_io_spec.md` is explicit:

> The native camera frame is `1920x1536`. The organizer directly squashes it to
> `224x224`; aspect ratio is deliberately not preserved and no padding/letterbox
> is added. **This matches the training-data conversion, which directly squashed
> native frames to square images.** Participants must not restore the native
> aspect ratio or add black bars before model inference.

We resize to **224x320** (`dataset/origami_dataset.py`, `F.interpolate(..., size=(224,320))`).
The robot never produces that shape.

The dataset videos are `480x480` — the same square squash the organizer applies,
just at higher resolution. So `480x480 -> 224x224` reproduces the deployment
transform exactly.

Measured discrepancy between what training sees and what deployment would feed:

| training target | deploy-vs-train image difference |
|---|---|
| **224x224** | **0.00 / 255** — the deploy image *is* the training image, zero resampling |
| 224x320 (current) | mean 0.92/255, **max 109/255**, concentrated on edges |

The max error lands on edges, which is exactly where the fold creases are.

Secondary benefit: 224x224 gives 49 tokens/camera (196 total) vs 70 (280) — a
**30% cut** on the transformer encoder, which runs **twice per training step**.

Where `224x320` appears (must stay in sync or train/inference diverge silently):
- `dataset/origami_dataset.py` — the live one
- `dataset/ha_pipelinev2_dataset.py:556` — old non-origami path
- `_inference.py:186` — old inference
- `orivizkar-submission-v000-gpu/vitac_policy_server.py:254` — `TRAIN_IMAGE_HW`

**Action:** change to 224x224 and lift it into `configs.py`. Deliberately not done
yet — it changes input geometry and would confound the four normalization runs.

Provenance, for the record: 224x320 is inherited from the *original* ViTacFormer
robot, whose head cameras were `180x320` (16:9) — see `dataset/data_tactile.py`
`camera_preprcess_info`. It was copied into the LeRobot origami path where the
cameras are square. There is no requirement behind it.

---

## 2. Observation history has no defined temporal spacing — **ASK** (highest risk)

The model consumes 6 state frames and 19 tactile readings, spaced `1/30 s` in
training. Each `infer` call delivers **one** timestep:
`observation/state (65,)`, `observation/tactile (60,)`.

The ring buffer in `orivizkar-submission-v000-gpu/vitac_policy_server.py` appends
one frame per `infer` call. So:

```
gap between buffered frames = (rows the organizer executed) / 30 s
```

`participant_zenoh_submission.md` says only:

> The organizer performs receding-horizon control and may consume only a prefix
> of the chunk. A later observation is authoritative.

The replan interval is **never stated**.

| organizer replans every | 6-step state history spans | tactile deltas |
|---|---|---|
| 1 step (30 Hz) | 0.17 s — as trained | correct |
| 5 steps | 0.83 s | **5x inflated** |
| 25 steps | 5.0 s | **25x inflated** |

Tactile is the casualty: half of `observation.tactile` is `diff()`, so those
channels would be 5-25x larger than anything seen in training. This is a worse
distribution shift than the image resize.

**Hard bound worth measuring:** the replan interval cannot be shorter than our own
inference latency. If the GPU forward takes 50 ms, calls cannot be closer than
~1.5 frames. **Measure GPU forward latency** — it tells us which row of that table
we are in.

Options, in the order I would do them:

1. **Ask the organizers** the replan interval / rows executed per chunk.
2. **Timestamp-resample.** Record `time.monotonic()` per observation, keep a longer
   buffer, resample onto the `1/30 s` grid. Identity at 30 Hz, correct
   interpolation below. The only approach right at *any* rate. Caveat: arrival
   time is not robot time (network + our latency), so it is an approximation.
3. **Reconstruct state history from our own previous chunk.** We returned 100
   absolute position targets; under position control the achieved state is close
   to the commanded row. Match the incoming `observation/state` to the nearest row
   of the previous chunk to recover index `k`, then use rows `k-5..k` — correctly
   spaced. Works for state, not tactile (though `tactile_hat` is literally a
   prediction of future tactile and could fill that gap).
4. **Frozen-history fallback.** If spacing is unknown, repeat the current
   observation rather than stacking wrongly-spaced frames. Deltas become zero,
   which the model *has* seen (every episode start, via LeRobot padding).
   Predictable degradation instead of a 25x scale error.
5. **Blunt:** retrain with `PROPRIOCEPTIVE_TEMPORAL_HORIZON=1` and a short tactile
   window. Removes the failure mode, costs whatever the history buys.

---

## 3. Submission server preprocessing is stale — **OPEN**

`orivizkar-submission-v000-gpu/vitac_policy_server.py`.

The ring buffer itself is **correct**: cold-start backfill matches LeRobot's
padding semantics (repeat first frame), `maxlen` 6 and 19 are right, ordering is
oldest->newest matching the *corrected* `configs.py`, tactile assembly
(`raw[1:]` + `diff` -> `[18,120]` -> `[1,2160]`) matches `nn.Linear(18*120, hidden)`,
the zero `tactile_next` placeholder is safe (unused at `epoch=999`), and `reset()`
clears both deques.

Three problems:

- **Misleading comment above `infer`.** It claims "no /255 scaling and no ImageNet
  mean/std normalization ... do not add either". `_preprocess_image` *does* divide
  by 255 (correct), so the comment contradicts the code and will mislead someone
  into removing it. ImageNet mean/std is now **required** — training applies it.
- **`TRAIN_IMAGE_HW = (224, 320)`.** Once §1 lands this becomes `(224,224)` and the
  `interpolate` is a no-op that should be deleted.
- **No input normalization / output denormalization.** Fine for an unnormalized
  checkpoint, breaks with normalized ones. Wire up `load_training_normalizer()`
  and the `normalizer_config.json` sidecar that training now writes.

---

## 4. The paper occupies ~1% of the frame — **DECIDE**

Measured on real frames (paper segmented as cream: `R-B > 18` and bright):

| frame | paper px | % of frame | bbox (native) | ResNet cells @224x224 |
|---|---|---|---|---|
| head_left t20 | 1189 | 0.52% | 270x284 | 3.9 x 4.1 |
| head_left t70 | 1941 | 0.84% | 275x254 | 4.0 x 3.7 |
| wrist_left t70 | 1095 | 0.48% | 100x186 | 1.5 x 2.7 |
| wrist_right t70 | 840 | 0.36% | 38x46 | **0.55 x 0.67** |

0.14-1.05% of the frame, and only ~1.6% of each bounding box is *visible* paper —
thin slivers between the grippers. Worst case it is smaller than a single feature
cell, i.e. under one of 49 tokens.

**We are hard-capped at 224x224 by the spec**, so this cannot be fixed with more
input resolution — training higher would itself be a train/deploy mismatch. The
available lever is **stride**, not input size:

| config | feat map | tok/cam | x4 | native px per token |
|---|---|---|---|---|
| resnet18 layer4 (current) | 7x7 | 49 | 196 | 69 |
| **resnet18 layer3 (stride 16)** | 14x14 | 196 | 784 | **34** |
| resnet50 layer4 + dilation | 14x14 | 196 | 784 | 34 |

Taking features from **layer3** gives 4x the spatial detail at the same input size
and backbone (`input_proj` becomes 256->hidden; position encoding and the
transformer adapt automatically). Cost: ~4x encoder compute, on a path traversed
twice per step.

**Trap:** `--dilation` is an exposed CLI arg but **crashes on resnet18** —
`NotImplementedError: Dilation > 1 not supported in BasicBlock`. It only works
from resnet50 up.

---

## 5. Why the old runs stalled at ~0.035 loss — **OPEN**

`0.035` is roughly the *"do nothing"* floor. Measured on held-out episodes, same
units as `l1`:

| predictor | L1 (rad) |
|---|---|
| predict the dataset mean | 0.1034 |
| constant-velocity extrapolation | 0.0697 |
| linear ridge on qpos history, **no vision** | 0.0504 |
| **copy the current pose, frozen 3.3 s** | **0.0439** |
| reported total loss (includes `10*kl + l1_tac`) | ~0.035 |

Actions are **absolute joint angles**: dataset std 0.759, but motion within a
100-step chunk is only 0.0333 — **96% of every target is "where the arm already
is"**, free to predict. The remaining 4% is the task. So the loss number carries
almost no information about task success, and improvements move it by ~0.001.

Contributing causes, ranked:

1. **Posterior collapse.** At total 0.035, `10*kl < 0.035` so `kl < 0.0035`; it is
   ~9-10 at init. `kl_weight=10` on a KL **summed over 32 latent dims** is very
   aggressive. Collapsed CVAE -> deterministic decoder -> conditional median of a
   multimodal distribution -> mode averaging. Check `train/kl_step`.
2. **LR decayed to ~0.** `get_cosine_schedule_with_warmup` decays to **exactly 0**;
   `lr_config['min_lr_ratio']` is a **dead key**, never passed anywhere. Repeated
   resumes keep advancing the schedule. Check `train/lr`; below ~1e-6 means the
   optimizer had stopped.
3. **Vision likely contributed little.** Images were unnormalized (fixed) *and*
   `lr_backbone=1e-5` is 30x below the main LR. A weak vision encoder degenerates
   the policy to proprioception-only — whose best predictor is roughly the copy
   baseline. Diagnostic: zero or shuffle the images at eval; if the loss barely
   moves, vision is being ignored.
4. **Arms dominate the gradient.** Copy-baseline error per group: `right_arm
   0.0705`, `left_arm 0.0613`, `right_hand 0.0424`, `left_hand 0.0400`, `motor
   0.0026`. Unnormalized L1 gives the arms most of the gradient; the fingers,
   where origami happens, get less.
5. **`l1_tac` was probably negligible** — unnormalized tactile std is ~0.005 in
   quiet stretches.
6. **Teacher-forcing cliff at epoch 75** — `tactile_pred` switches from ground
   truth to self-predicted. Only relevant if training got that far.

**A deeper fix worth considering: predict delta actions.** `action(t+k) - state(t)`
makes the target the 0.033 of real motion instead of 0.759 of mostly-current-pose.
Same information, but the loss would be *about* the motion. `_inference.py` already
does `abs_action = action + ref_qpos`, so an earlier version of this pipeline was
relative — worth finding out why that changed.

---

## 6. Tactile value/delta scale imbalance — **OPEN, low priority**

`observation.tactile` is `concat(values, diff(values))` through a single
`input_proj_tactile`. Whole-dataset pooled ratio: **16x raw, 39x normalized**.

Per channel the ratio is **identical** either way — normalization is a per-dim
linear map, so `diff(x/s) = diff(x)/s` cancels exactly (verified: `left_thumb_fz`
20.3 vs 20.3). The pooled shift is only a change in which channels dominate.

So the 20x gap is **intrinsic to the signal** — at 30 fps a tactile reading barely
changes between frames relative to its range. Not a normalization artifact. Worth
an experiment (scale the delta half, or give it its own projection), not urgent.

*(An earlier note in this project claimed 1400x. That came from a pooled std over a
2-sample batch and was wrong. The log line now reports a median per-channel ratio.)*

---

## 7. Loss coefficients were never swept — **OPEN**

`loss = l1 + 10*kl + 1.0*l1_tac`. The three terms have never been balanced against
each other. Both are now flags (`--kl_weight`, `--tac_weight`) and all three terms
are logged separately at TRACE. Given §5.1, `kl_weight` is the first to try
lowering.

---

## 8. Normalizer statistics include the validation episodes — **OPEN, minor**

Normalization is built from `meta/stats.json`, which aggregates **all** episodes
including the 2 held out. With a 2-episode holdout the effect is negligible, but
it is a leak. A clean fix computes stats over the train episodes only.

---

## 9. Never exercised — **UNTESTED**

No GPU on the dev box, so these have not been run even once:

- **DDP / multi-GPU** — `accelerator.prepare`, gradient sync, the barrier in the
  backbone loader
- **bf16 / fp16 autocast** — including whether the loss terms behave under
  autocast
- **The accelerate config matrix** — all 12 files verified by inspection and the
  launcher asserts the precision matches, but none has actually been launched
- **The full submission server against a live evaluator**

Watch the first job's startup lines
(`accelerator: num_processes=4 ... mixed_precision=bf16`) before queueing more.

Also worth checking on the server: `run_ori_job.sh` does
`cd $HOME/other/ori/ori/ViTacFormer`. All work is in `vitacformer++`. Confirm that
path resolves to the tree with the fixes:

```bash
grep -c "action_is_pad" $HOME/other/ori/ori/ViTacFormer/dataset/origami_dataset.py
# 0  -> OLD tree, none of the fixes are running
```

And the scratch script copies the **entire** dataset regardless of
`MAX_EPISODES`, so a 48-episode run may still copy hundreds of GB to NVMe.

---

## Confirmed compatible with the spec

For the record, these were checked against `robot_io_spec.md` and match:

- action layout `0:7 / 7:29 / 29:36 / 36:58 / 58:65` — identical to `JOINT_GROUPS`
- absolute joint angles in radians, `float32[T,65]`, finite
- `action_horizon` = `CHUNK_SIZE` = 100, within the allowed `[1,1024]`
- tactile `(60,)` = 10 fingers x `[fx,fy,fz,tx,ty,tz]` — this also **confirms** the
  `i % 6` axis grouping used for shared-scale tactile normalization
- "must denormalize ... before replying" — implemented
- `joint_torque[58:65]` always zero — matches `stats.json`; not consumed
- ignoring `tactile_deform` / `tactile_raw` is an explicit participant choice

## Resolved

Kept short; see the git log for detail.

- **B1** tactile/state `delta_timestamps` were newest-first: history reversed,
  current frame dropped, one delta spanning 0.63 s
- **B2** `action_is_pad` discarded — at frame `n-60`, 40 of 100 actions were
  fabricated and trained at full weight; loss denominator also counted padding
- **B3** `MIXED_PRECISION=fp32` silently ran bf16 on 4 GPUs, fp16 on 1 and 8
- **B4** `global_step` restored on resume then immediately reset to 0
- **B5** `DATASET_ROOT` unset raised an opaque `Path(None)` `TypeError`
- **B6** images fed raw `[0,1]` to an ImageNet-pretrained ResNet
- **B7** normalizer stats cast to bf16 — `action[58]`'s `q01`/`q99` collapse to the
  same value in bf16 **and** fp16, making the denominator exactly 0
- **B8** dims 58/59 patched on `action` only, not `observation.state`
- **B10** dead `joint_torque` work on the per-batch critical path
- tactile per-dim normalization amplified idle-finger noise: worst normalized
  value **135 -> 3.20** after sharing one scale per axis across the 10 sensors
- no validation split at all; `min_val_loss` was permanently `inf`
- validation metrics were in normalized units, incomparable across runs
- inference never applied the training normalizer nor denormalized its output
- backbone loaded ImageNet weights on rank 0 only

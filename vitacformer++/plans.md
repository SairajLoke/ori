# Plans

Deferred work with enough detail to pick back up cold. Not urgent, not started.
See `concerns.md` for open bugs/risks in what's already built; this file is for
what's intentionally not built yet.

---

## 1. Integrate joint_torque

Currently dead: `configs.DELTA_TIMESTAMPS` has no entry for it, so it isn't
even pulled with a time window; `convert_batch` computes it only for a TRACE
debug log and never puts it in `output`. Confirmed via
`robot_io_spec.md` + the actual stats that `joint_torque[58:65]` (the motor
block) is hard-zero from the robot, not a data artifact -- whatever plan we
use has to tolerate 7 constant channels without special-casing them.

**Design decision, already made:** piggyback on the existing proprio token
(concatenate onto `qpos` along the feature dim, same
`PROPRIOCEPTIVE_TEMPORAL_HORIZON` window) rather than giving it a dedicated
token like tactile has. A dedicated token means rewriting
`transformer.py`'s hardcoded token-slice cross-attention boundaries
(`src[:2]`, `src[2:4]`, `src[4:]` etc, sized for 1-2 "extra" tokens today) --
real regression risk for a signal that's just co-sampled proprioception, not
a distinct modality the way tactile (own future-prediction/teacher-forcing
machinery) is.

Concrete flow, file by file:

1. `configs.py`: `DELTA_TIMESTAMPS["observation.state.joint_torque"] =
   DELTA_TIMESTAMPS["observation.state"]` (reuse the same window).
2. `dataset/origami_dataset.py` `convert_batch`: normalize it (mirror the
   `lowdim`/`action` pattern already there), `torch.cat([lowdim, joint_torque],
   dim=-1)` before it enters `output`. Delete the current TRACE-only dead
   branch.
3. `my_utils/normalizer.py` `recommended_modes()`: add
   `"observation.state.joint_torque": "quantile"`. No other normalizer change
   -- the degenerate-spread guard (already built) automatically pins the 7
   always-zero motor-torque dims to unit scale instead of amplifying noise,
   the same way it already handles action/state dims 58-59.
4. `detr/models/detr_vae.py`: `qpos_dim = (state_dim + torque_dim) *
   proprioceptive_temporal_horizon`. `input_proj_robot_state` and
   `encoder_joint_proj` both read `qpos_dim` already -- resize both together,
   easy to update one and miss the other.
5. `origami_imitate_episodes.py` MASK_FINGERS: existing `apply_joint_mask`
   calls target the *state* portion's indices and are unaffected as long as
   torque is appended AFTER state, not interleaved (confirmed by tracing the
   indexing). Optional: two more `apply_joint_mask` calls at
   `start_index=65+7` and `65+35` to mask torque for disabled fingers the same
   way state is masked. Off by default anyway (`MASK_FINGERS=False`).
6. Logging: swap the dead TRACE line for the same before/after normalization
   logging `lowdim`/`action` already get.

Explicitly out of scope for the first pass:
- **Auxiliary future-torque prediction** (a `torque_hat` head + teacher
  forcing, mirroring tactile's `tactile_next`). Torque is contact feedback,
  so this is a legitimate later idea, but it's dedicated-token-sized work,
  not a "just wire it in" change.
- **Dropping the 7 known-zero motor dims** rather than feeding them through
  and letting the degenerate guard neutralize them. Keeping them is simpler
  (uniform code path) and costs nothing (7 constant channels are trivial for
  the network to ignore). Revisit only if trimming params later matters.

Verification plan when this gets built: shape check (`qpos_dim` old vs new,
both `input_proj_robot_state` and `encoder_joint_proj` resized), a real
forward+backward+optimizer step confirming gradient reaches the new slice,
the normalizer's degenerate-dim log confirming it caught the 7 zero
motor-torque dims, full CPU smoke run end to end -- same standard as the ViT
backbone work.

---

## 2. Stage / phase predictor

Origami folding is sequential and repetitive -- the same visual state can
recur at different points in the task with a different correct next action,
which is exactly the failure mode chunked BC with a CVAE latent
(mode-averaging) is prone to on repetitive multi-step tasks. A stage label
disambiguates.

**Labelling cost is the whole reason to do this before item 3.** A few hours
of manual segment boundaries per episode, or semi-automatic from gripper
open/close events + arm-velocity minima, gets `[T]` integer stage labels.
Compare to building and validating a keypoint tracker (item 3) -- labelling
cost here is near zero.

Recommended design:
- Use it as BOTH an auxiliary classification target AND a conditioning input
  -- same two-pass predict-then-condition pattern tactile already uses
  (`tactile_hat` predicted in pass 1, fed into pass 2 via
  `additional_pos_embed`/teacher forcing). Predicting-only would give a
  representation-shaping gradient; conditioning is what actually resolves the
  visual-state ambiguity that causes mode-averaging.
- At inference the stage must be self-predicted (no ground truth available),
  so a misclassification cascades into wrong actions. Condition on the
  SOFTMAX-weighted average of stage embeddings, not a hard argmax lookup, so
  a 60/40 prediction degrades gracefully instead of committing hard to a
  wrong stage.
- Gives per-stage validation metrics for free once built -- makes every other
  experiment on this list more interpretable ("the ViT backbone helped on
  stage 3 only", etc).

Not designed in file-level detail yet -- do that pass when this is picked up,
likely mirroring the tactile token/head pattern in detr_vae.py + transformer.py
(the exact hardcoded-token-slice risk flagged in item 1 above becomes
unavoidable here, since this DOES need its own token).

---

## 3. Paper keypoint predictor

Track keypoints on the paper through an episode (sampled along edges, per the
ACT/ALOHA-style papers that motivate this), predict their next-state
positions as an auxiliary task -- forces the vision encoder to represent
paper GEOMETRY rather than only arm pose, which a pure BC objective on joint
angles never explicitly asks for.

**The hard part is track quality, not the model.** Paper is textureless,
self-occluding, and specular under this lighting -- a tracker (CoTracker3,
DINO-tracker, etc) will drift on blank fold faces and lose points under a
hand. Measured earlier (see the paper-occupancy work in concerns.md #4): the
paper is 0.14-1.05% of the frame and often smaller than one backbone feature
cell, which is exactly the regime trackers struggle in too.

**Before writing any tracking code:** run a tracker on 2-3 real episodes
offline and look at the overlays. If tracks survive under ~50% of an episode,
the target is noise and will hurt more than it helps. This is a go/no-go
gate, not optional groundwork.

Design notes, if the gate passes:
- **Which points, ranked by robustness (most robust first):**
  1. Grid of points on the paper mask (from a cheap color/SAM mask),
     resampled per episode -- dense, redundant, tolerant of individual track
     loss. Add a per-point VISIBILITY flag; mask the loss by it (a
     confidence-reporting tracker gives this near-free). Without visibility,
     the model gets trained to predict hallucinated positions for occluded
     points.
  2. Detected line segments / crease lines instead of points -- more stable
     on textureless regions than point features.
  3. Explicit corner keypoints -- best signal when visible, worst
     reliability (corners are exactly what disappears under the hand mid-fold).
  Start with (1).
- **Head cameras only.** Wrist cameras move WITH the hand, so tracking there
  conflates paper motion with camera motion -- a much harder target than
  tracking in a head view where only the paper moves.
- **Precompute offline, once per episode** -- run the tracker once, write
  `[T, K, 3]` (x, y, visible) as a new LeRobot dataset feature. Then it flows
  through `delta_timestamps` exactly like `observation.tactile` already does
  -- no per-step tracker cost during training, `convert_batch` changes by
  ~5 lines once the feature exists.
- **Architecturally this is the tactile pattern again**: extra query
  embeddings + a linear head + an L1 term against the tracked (x,y). The
  two-pass predict-then-condition machinery already built for tactile is
  directly reusable here, which is part of why item 2 (stage predictor)
  should land first -- by the time this is built there will be two working
  examples of that pattern to copy instead of one.

Sequencing note (unchanged from the original roadmap discussion): item 2 before
item 3. Cheaper labels, fixes a known BC failure mode, and produces a second
working instance of the predict-then-condition pattern before attempting it a
third time here.

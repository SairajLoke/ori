# Chunk-seam smoothing at inference

Why the assembled trajectory jumps at chunk boundaries, what the contract does and
does not let us do about it, what was implemented, and what the numbers say.

---

## 1. The problem

Each `infer()` call is an independent rollout. Two consecutive chunks are never
constrained to agree about the timesteps they share, so the trajectory the
organizer assembles can step discontinuously at every seam even though each chunk
is individually smooth.

Measured on an earlier real checkpoint (`preds_50.npy`, 50 queries x 25 x 65,
`frame_stride == action_horizon == 25`):

| | mean per-step delta |
|---|---|
| within a chunk | 0.00205 rad |
| **at a seam** | **0.0569 rad (28x)**, max **1.08 rad (~62 deg)** |
| ground truth, same indices | 0.00102 rad (no elevation at all) |

Ground truth is just as smooth across those boundaries as anywhere else, so this
is a model artifact, not a property of the task. The worst offenders were finger
joints (ring / middle / pinky PIP / DIP).

## 2. What the docs actually say

**Cadence is never specified.** Nothing in `docs/` states a rate, period, or Hz
for `infer()`. What is stated:

- `participant_zenoh_submission.md:243` — "The organizer performs receding-horizon
  control and **may consume only a prefix** of the chunk. A later observation is
  authoritative; the service must not assume that all previously returned rows
  were executed."
- `competition_participant_complete_guide.md:497` — "The organizer **may execute
  only the first few steps** before running inference again, or may perform an
  open-loop rollout over multiple chunks in Shadow to produce 100 future steps."
- `container_submission.md:112` — "The organizer may consume only a prefix and
  then replan; the policy may not publish, schedule, or assume execution of its
  result."

So consumption depth per call is organizer-controlled and unknown, and may vary
call to call. Any fix must work without knowing it.

**There is no dense history.** `robot_io_spec.md` §1 defines one observation per
`infer()` call: four `(224,224,3)` frames, one `(65,)` state, one `(60,)` tactile
vector. No stacks, no past actions, no timestamps. The 6-step state window and
19-step tactile window the model trains on have to be rebuilt server-side across
successive calls — which means **the history window's real time span is set by the
call cadence, not by us**. At 30 Hz calls the 6-step window spans 0.17 s, matching
training; at 25-step-apart calls it spans 5 s, and the tactile delta channel (a
`torch.diff`, whose magnitude scales with elapsed time) is mis-scaled by the same
factor. This is already addressed on the training side in `vitacformer++`
(`ORI_USE_OBS_FPS`, `ORI_OBS_FPS`, `ORI_JITTER_HISTORY`); the two checkpoints
compared below are exactly that pair.

**Smoothing is explicitly ours to do.** `robot_io_spec.md` §6 — "History, image
normalization, modality selection, language tokenization, diffusion sampling and
**temporal ensembling are participant-internal**. Any episode-scoped state must be
cleared by `reset`." Nothing in the contract says the organizer smooths anything.

**And it is checked.** `remote_participant_development.md:192` — when the official
URDF parses, the evaluator "checks position, **jumps between adjacent steps**, and
velocity". So a seam jump is a compatibility-check risk, not a cosmetic one.

**Constraints that bound any fix:** reply must stay `float32[T,65]`, finite, `T`
fixed for process lifetime and equal to metadata `action_horizon`, absolute joint
radians in the §2 column order, and every episode-scoped buffer cleared on `reset`.

### The Shadow rollout rule matters

`remote_participant_development.md:160` — "If a model's horizon is shorter than
100, the evaluator performs multiple open-loop chunks. Each new chunk uses the
**final step of the previous chunk** as the local `observation/state`, while
images and the prompt remain from the same remote snapshot."

That is a feedback loop through our own output, and it is what the `shadow` regime
in the evaluator below reproduces.

## 3. Approaches

Three mechanisms, all contract-legal, selected by one flag. They compose, so two
combined modes are also exposed.

**`anchor` — re-tie the chunk to the measured pose.** The one value the contract
calls authoritative is the incoming `observation/state`. `chunk[0]` is the model's
estimate of *now*, so the first row we can actually command is `chunk[1]`: the plan
is advanced one step, and the residual offset `state - chunk[0]` is added back and
decayed to zero over `blend_steps`. Row 0 lands on `state + (chunk[1]-chunk[0])` —
measured pose plus the model's own first-step velocity.

> Emitting `state` itself would be zero-jump but zero-progress, and under the
> Shadow rule (next state := final row of previous chunk) a stride-1 caller feeds
> that pose straight back, freezing the robot. The one-step advance is what avoids
> that fixed point; it was a real bug caught by the stride-1 shadow test.

**`ensemble` — average the overlap with the previous chunk.** ACT-style temporal
ensembling, named as allowed in §6. We are never told how many rows were executed,
so it is recovered: the row of the previous chunk closest to the measured pose is
where the robot got to. The overlap is then blended with exponentially increasing
trust in the new chunk. In the Shadow regime this recovery is exact.

**`clamp` — rate-limit per-step deltas.** Walk forward from the measured pose
capping every joint delta at `max_step_rad`. Bounds the evaluator's adjacent-step
jump/velocity check by construction, regardless of what the model emitted.

**`auto` — dispatch on whether overlap survived.** Added after §6.2 showed no
fixed mode is safe at every consumption depth: `ensemble` has nothing to average
once the caller drains the whole horizon, and `anchor` under-moves when applied
every single step. `auto` uses `ensemble` when previous-chunk rows still overlap
and `anchor` when they do not, then clamps. Since cadence is unknowable, this is
the only mode that cannot be defeated by the organizer's choice of stride.

## 4. What changed

| file | change |
|---|---|
| `smoothing.py` | **new** — `anchor` / `ensemble` / `rate_limit` / `smooth_chunk`, pure numpy, no side effects |
| `vitac_policy_server.py` | flag + params in `TeamPolicy.__init__`; `_prev_chunk` buffer cleared in `reset()`; one `smooth_chunk()` call at the end of `infer()` |
| `Dockerfile` | copy `smoothing.py`; add `/app` to `PYTHONPATH` |
| `scripts/eval_smoothing.py` | **new** — offline receding-horizon replay evaluator |
| `scripts/render_smoothing_report.py` | **new** — renders the tables below from the result JSON |
| `scripts/local_contract_test.sh` | forward `VITAC_SMOOTHING` / `VITAC_OPTIMIZATION` into the container |

Selected like the existing optimization flag, with an env override:

```python
SMOOTHING_IDX = 6                      # 'auto' -- index into smoothing.MODES
self.smoothing = os.environ.get("VITAC_SMOOTHING") or SMOOTHING_MODES[SMOOTHING_IDX]
```

```
VITAC_SMOOTHING     none | anchor | ensemble | clamp | anchor_clamp | ensemble_clamp | auto
VITAC_BLEND_STEPS     anchor decay length, default 4
VITAC_ENSEMBLE_DECAY  ensemble trust ramp, default 0.35
VITAC_MAX_STEP_RAD    clamp limit, default 0.10
VITAC_OPTIMIZATION    compile | tflite | none  (default compile; see §8)
```

`none` is an exact identity passthrough, so the flag is a true A/B.

Unit-checked: every mode returns finite `float32[T,65]`, is safe on the first call
after `reset()` (`prev_chunk is None`), and `clamp` provably respects
`max_step_rad`.

## 5. Evaluation method

`scripts/eval_smoothing.py` replays held-out **episode 0** (a `val_episodes` entry
in both training configs) from
`lerobot3.0_shortgop15_224` through the real checkpoint, one observation per call,
rebuilding history exactly as `TeamPolicy` does, consuming only `stride` rows
before re-querying. It imports the shipped `smoothing.py`, so it exercises the
same code the server runs.

Two regimes, because they answer different questions:

- **`dataset`** — each call observes the episode. The robot effectively ignores our
  commands, so a mode that anchors to the measured pose is *penalised* whenever our
  trajectory has legitimately drifted from ground truth. Fair for fidelity (`MSE`),
  pessimistic for seam metrics.
- **`shadow`** — images/tactile frozen at the first frame, state fed from our own
  last executed row. This is the documented evaluator behaviour for horizon < 100,
  and the fair test for continuity.

Metrics: `seam` = mean |x[t+1]-x[t]| at chunk boundaries; `max_step` = worst
adjacent-step delta anywhere (evaluator jump-check proxy); `MSE` = assembled
trajectory vs dataset ground-truth actions. Plus a stride- and mode-independent
measure of the raw artifact — how much consecutive chunks disagree about the
timesteps they share — and open-loop error vs horizon depth.

## 6. Results

Two checkpoints, both epoch 70 on the same data, differing only in how the
observation history was sampled during training:

- **base** — `aug20_resnet18_224_...065113`, history windows at FPS=30
- **jitter** — `aug20_resnet18_224_jitter_obsfps5_...065144`, `ORI_OBS_FPS=5` with
  `ORI_JITTER_HISTORY=1` (gaps jittered up to 3x)

### 6.1 The artifact scales with consumption depth, not with anything we control

Raw disagreement between consecutive chunks about the timesteps they share —
no smoothing involved, so this is the size of the problem itself:

| stride | base (dataset) | base (shadow) | jitter (dataset) |
|---|---|---|---|
| 1 | 0.0049 | 0.0044 | 0.0057 |
| 5 | 0.0169 | 0.0257 | 0.0220 |
| 25 | **0.1157** | **0.1948** | **0.0753** |

Disagreement grows ~24x from stride 1 to stride 25. **Re-query cadence is the
single biggest lever, larger than any post-hoc smoothing.** This is the part of
Real-Time Chunking that transfers to a non-diffusion policy: the fix for seams is
mostly to not run 25 steps open-loop.

Note the last column: the jitter checkpoint's disagreement at stride 25 is **35%
lower than base**, i.e. the training-side observation-cadence fix attacks the same
artifact at its source.

### 6.2 Shadow regime — the documented evaluator behaviour

`base`, held-out episode 0, horizon 25. `path` vs `gt_path=44.8` catches modes that
look smooth only because they stopped moving.

| stride | mode | seam | max_step | MSE | path |
|---|---|---|---|---|---|
| 25 | `none` | 0.18046 | 1.1645 | 0.16360 | 102.8 |
| 25 | `clamp` | 0.06787 | 0.1000 | 0.14917 | 99.6 |
| 25 | `ensemble` | 0.18046 | 1.1645 | 0.16360 | 102.8 |
| 25 | `anchor_clamp` | 0.00336 | 0.1000 | **0.13310** | 96.5 |
| 25 | **`auto`** | **0.00293** | 0.1000 | 0.13482 | 97.2 |
| 5 | `none` | 0.01608 | 0.1912 | 0.12403 | 70.2 |
| 5 | `anchor` | 0.01015 | 0.2376 | **0.07915** | 52.9 |
| 5 | **`auto`** | 0.00988 | 0.1000 | 0.09400 | 48.5 |
| 1 | `none` | 0.00750 | 0.2097 | 0.07311 | 48.3 |
| 1 | `anchor_clamp` | 0.00449 | 0.0625 | 0.20264 | **28.9** |
| 1 | **`auto`** | 0.00831 | 0.1000 | **0.04488** | 53.5 |

At the deployed stride 25, `auto` cuts the seam **62x** (0.180 → 0.0029), bounds
`max_step` at 0.1 rad, and improves MSE 18%.

Two failure modes the table exposes:

- **`ensemble` is a no-op at stride == horizon** (identical to `none`). Once the
  caller consumes the whole chunk there is no overlap left to average against.
  This is correct behaviour, not a bug — but it means ensemble alone cannot be the
  answer for a caller that drains the horizon.
- **`anchor_clamp` collapses at stride 1**: MSE 0.203 (worst in the table) with
  `path` 28.9 against a ground-truth path of 44.8 — it is under-moving by a third.
  Anchoring every single step turns the rollout into pure velocity integration,
  which drifts.

`auto` exists precisely because of those two rows: it dispatches to `ensemble`
when overlap remains and to `anchor` when it does not, then clamps. It is the only
mode that is at or near best across all three cadences, and never pathological.

### 6.3 Dataset regime — fidelity, and a caution about anchoring

Here observations come from the episode regardless of what we command, so anchoring
is penalised whenever our trajectory has legitimately drifted from ground truth:

| ckpt | stride | mode | seam | MSE | path (gt 44.8) |
|---|---|---|---|---|---|
| base | 25 | `none` | 0.10769 | 0.03604 | 80.4 |
| base | 25 | `anchor_clamp` | 0.08257 | **0.02689** | 84.3 |
| base | 25 | `auto` | **0.06042** | 0.02839 | 80.6 |
| base | 5 | `anchor` | 0.05743 | 0.00472 | **173.2** |
| base | 5 | `auto` | 0.02018 | 0.00810 | 96.3 |
| jitter | 25 | `none` | **0.06334** | **0.01798** | 73.3 |

Two things to take from this:

- `anchor` at stride 5 produces a **173.2** path length against a ground truth of
  44.8 — a sawtooth, because it snaps to the measured pose each query while the
  plan diverges. That is an artifact of this regime (a real robot tracks our
  commands), but it is the reason `path_len` is reported at all.
- The **jitter checkpoint with no smoothing at all** (seam 0.063, MSE 0.018) beats
  the base checkpoint with the best smoothing (seam 0.083, MSE 0.027). Fixing the
  observation cadence in training does more than fixing the seam at inference.

### 6.4 How deep is a chunk worth consuming?

Open-loop MSE vs horizon depth, averaged over 12 query points:

| depth | 1 | 5 | 10 | 20 | 25 | 40 | 50 | 75 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.0135 | 0.0139 | 0.0121 | **0.0106** | 0.0151 | 0.0282 | 0.0299 | 0.0278 | 0.0164 |
| jitter | **0.0082** | 0.0110 | 0.0151 | 0.0289 | 0.0379 | 0.0409 | 0.0355 | 0.0181 | 0.0104 |

`base` is flat to depth ~20 then roughly triples by depth 40-50 — so consuming
much past ~20-25 rows is where the prediction itself stops being worth trusting,
independent of seams. `jitter` is sharper near-term (d1 0.0082 vs 0.0135) and
degrades faster with depth, consistent with being trained on 5 Hz observation
windows. Both argue for shallow consumption; neither supports draining a 100-row
chunk.

## 7. Recommendation

1. **Ship `auto`** (now the default, `SMOOTHING_IDX = 6`). Cadence is
   organizer-controlled and unknown, and §6.2 shows no fixed mode is safe across
   the range — `ensemble` degenerates at full-horizon consumption, `anchor_clamp`
   under-moves at stride 1. `auto` bounds `max_step` at 0.1 rad in every case,
   which is the evaluator's adjacent-step check.
2. **Prefer the jitter checkpoint.** It has 35% lower raw chunk disagreement and
   half the MSE at stride 25 before any smoothing is applied.
3. **Smoothing is second-order to cadence.** Disagreement grows 24x from stride 1
   to 25; if the query budget allows more frequent `infer()` calls, that is worth
   more than any mode here. At the measured GPU latency (~50 ms against 833 ms of
   chunk playback) there is ample headroom.
4. Do not raise `action_horizon` above 25 — §6.4 shows prediction quality falls off
   past depth ~20-25 on both checkpoints.

## 8. Contract validation

One command runs the whole thing — build (cached), router, policy, both validator
passes — and leaves the policy container up with its log streaming:

```bash
./scripts/contract_local.sh          # build + synthetic + dataset checks
./scripts/contract_local.sh logs     # follow the policy log
./scripts/contract_local.sh down     # tear down
```

It is entirely local: a bridge network, the Zenoh router published only on
`127.0.0.1:17447`, and the policy container reaching it by container name. No GPU
required, nothing leaves the host. `torch.compile` is off by default there
(`VITAC_OPTIMIZATION=none`) because on a CPU-only box the inductor cache is slow
and can exhaust RAM mid-startup. One fixed image tag, containers force-removed by
name, and the superseded image ID deleted after a rebuild, so repeated runs do not
accumulate copies. It also refuses `--obs-type dataset` on a root missing
`tactile_deform` instead of failing halfway through (see below).

`check_zenoh_policy.py` against the built image with `VITAC_SMOOTHING=auto`:

```
metadata: PASS (transport=origami-zenoh-v1, semantic=origami-v1, horizon=25, dim=65)
reset:    PASS
infer Synthetic frames 1..6: PASS  (~280 ms, incl. "without optional tactile_raw")
PASS: policy is compatible with origami-zenoh-v1        --obs-type synthetic

infer real-dataset 1/6: PASS (340.6 ms, full, ep=0 frame=0   t=0.000s)
infer real-dataset 2/6: PASS (493.0 ms, full, ep=0 frame=25  t=0.833s)
...
infer real-dataset 6/6: PASS (286.1 ms, without optional tactile_raw, ep=0 frame=125)
PASS: policy is compatible with origami-zenoh-v1        --obs-type dataset
```

Both runs used the untrained checkpoint in `checkpoints/`, so the per-query `loss`
the validator prints is meaningless — this exercises the transport, schema, and
smoothing path, not model quality. Latencies are CPU-only.

Two environment issues surfaced, neither caused by the smoothing change:

- **`torch.compile` on a CPU-only host** exhausts RAM and kills the container
  during startup. `VITAC_OPTIMIZATION` is now an env override (same pattern as
  `VITAC_SMOOTHING`) so the image can run with compile disabled for local
  validation; the default is still `compile`. `scripts/local_contract_test.sh`
  forwards both variables into the container.
- **`lerobot3.0_shortgop15_224` cannot drive the dataset replay.** Its
  `meta/info.json` lists only the four RGB cameras — `tactile_deform` and
  `tactile_raw` were dropped when that root was built (cf. the training root name
  `..._notactimg`). `robot_io_spec.md` §1 marks `observation/image/tactile_deform`
  **required** (only `tactile_raw` is optional), so the validator cannot assemble a
  full observation and `infer` fails schema validation before the model runs.
  Use a root that still carries the tactile stream (e.g. `lerobot3.0_shortgop15`)
  for `--obs-type dataset`, or `--obs-type synthetic`.

## 9. Reproducing

```bash
PYTHONPATH=.:vitacformer:vitacformer/detr \
python scripts/eval_smoothing.py \
  --dataset_root .../lerobot3.0_shortgop15_224 \
  --ckpt_dir .../assets/aug20_resnet18_224_jitter_obsfps5_..._ori_tactile \
  --episode 0 --total_steps 100 --strides 1,5,25 --regime shadow \
  --out eval_results/jitter_shadow.json

python scripts/render_smoothing_report.py eval_results   # tables above
```

Raw results: `eval_results/{base,jitter}_{dataset,shadow}.json`.

## 10. Caveats

- Numbers come from **one held-out episode** (episode 0, a `val_episodes` entry in
  both training configs) of one season. Directionally consistent across two
  checkpoints and two regimes, but not a multi-episode benchmark.
- Evaluation ran **CPU-only** (no NVIDIA container runtime on this host), so the
  ~260 ms/query figures are not deployment latency.
- `shadow` MSE is inflated for every mode because images stay frozen at the first
  frame for the whole rollout, by design of that regime. Compare modes within a
  regime, not across regimes.
- The `clamp` limit (`VITAC_MAX_STEP_RAD=0.10`) was not tuned against the official
  URDF velocity limits; it is a conservative default that comfortably exceeds
  observed ground-truth per-step motion (mean 0.0012 rad). Worth re-deriving from
  the URDF before the real run.


# Experiment runbook

Everything new is a flag, all defaults preserve current behaviour, and every flag
reaches the trainer through `EXTRA_ARGS` (appended verbatim by
`origami_train_multigpu.sh`) — the run scripts need no edits.

Dataset assumed: **pre-resized 224, shortgop, NOT pre-cropped**. `--image_crop`
crops at training time, so one dataset serves every crop ablation.

---

## 0. Before any run: regenerate the weights on YOUR data

`action_weights.json` in the repo was derived from **14 episodes of one season**
on the laptop. Torque magnitudes, cross-episode CV, the delta spread and the
constant-dim detection will all differ on 500 episodes. Regenerate once:

```bash
cd $HOME/other/ori/ori/vitacformer++
python tools/compute_action_weights.py \
  --dataset_root $DATASET_SRC \
  --out action_weights.json
```

Sanity-check the output before trusting it:

- finger weights should keep the order **thumb > index > middle >> ring ~ pinky**
- `constant_dims` should contain **58, 59** (auto) and **64** (forced, `-0.8727`)
- `delta_stats.floored_dims` should be a handful (7/65 locally), not most of them

If ring/pinky do **not** come out lowest on the full dataset, stop and tell me —
the whole weighting rationale rests on that separation.

---

## 1. Speed / memory checks (no training, minutes)

```bash
# does fused attention actually fire, and what does it save?
python tools/check_sdpa.py --batch 64 --seq 200 --dtype bfloat16
python tools/check_sdpa.py --batch 96 --seq 200 --dtype bfloat16
```

Read the `attention kernels` column: `FUSED: [...flash...]` or `mem_efficient`
means it is really being used; `unfused: [bmm, softmax]` means it is not. In fp32
you will often get `mem_efficient` rather than flash — still fused, still no
materialised `[T,T]` matrices. You train in bf16, so flash should be reachable.

---

## 2. The gate — run this first, nothing else matters until it passes

The model currently sits at **parity with "hold current pose"** (R2 -0.157;
-0.53 on right_hand). Every loss change below assumes the mapping is learnable.

```bash
# E0  baseline, unchanged behaviour, for reference
EXPT_NAME=e0_baseline USE_SCRATCH=1 MIXED_PRECISION=bf16 USE_NORMALIZATION=1 \
phd run -ng 4 -p shr_gpu -GR H100 -l %J.log sh ori/ori/run_ori_job.sh 4
```

**The number to read** is `val_l1/all_dims` in the log — mean |error| per dim in
radians, unweighted, so it is comparable across every variant below. Reference
points already recorded in the code:

| predictor | val_l1/all_dims |
|---|---|
| copy current pose, frozen 3.3s | **0.044** |
| predict the dataset mean | 0.103 |

**If no variant gets clearly below 0.044, the model has not beaten "do nothing"**
and the remaining work is data/architecture, not loss shaping.

---

## 3. Loss experiments (the substance) — one variable at a time

```bash
W="--loss_dim_weight_mode file --action_weights_json action_weights.json"

# E1  torque-derived per-dim weights
EXPT_NAME=e1_weighted EXTRA_ARGS="$W" \
phd run -ng 4 -p shr_gpu -GR H100 -l %J.log sh ori/ori/run_ori_job.sh 4

# E2  relative actions (predict pose delta, not absolute)
EXPT_NAME=e2_deltas EXTRA_ARGS="$W --predict_deltas" \
phd run -ng 4 -p shr_gpu -GR H100 -l %J.log sh ori/ori/run_ori_job.sh 4

# E3  + temporal weighting (only 25 of 100 rows are ever returned)
EXPT_NAME=e3_deltas_temporal \
EXTRA_ARGS="$W --predict_deltas --temporal_weight_mode horizon --action_horizon_for_weights 25" \
phd run -ng 4 -p shr_gpu -GR H100 -l %J.log sh ori/ori/run_ori_job.sh 4

# E4  lower-dimensional output: 45 dims, ring/pinky held at the measured pose
EXPT_NAME=e4_active45 EXTRA_ARGS="$W --action_dims_mode active" \
phd run -ng 4 -p shr_gpu -GR H100 -l %J.log sh ori/ori/run_ori_job.sh 4
```

E2 is the one with a mechanism behind it: it removes the "copy qpos" term that
currently dominates the loss. E1/E3 reshape weighting. E4 also removes ring/pinky
**actuation**, so treat it as a behaviour change, not just a loss change.

---

## 4. Compute experiments — validate on SHORT runs, then fold into the winner

These should not change quality much; run 2-3 epochs and read **s/step** and peak
memory, not val loss.

```bash
# E5  tactile as a plain input token: ONE transformer pass instead of two
EXPT_NAME=e5_tactile_input EXTRA_ARGS="$W --tactile_mode input" ...

# E6  fused attention (need_weights=False)
EXPT_NAME=e6_flash EXTRA_ARGS="$W --explicit_flash_attn" ...

# E7  centre-crop 224 -> 192: 49 -> 36 tokens/cam
EXPT_NAME=e7_crop192 EXTRA_ARGS="$W --image_crop 192" ...

# E8  drop one head camera (each ablates at -1.4%/-1.9%)
EXPT_NAME=e8_cam3 EXTRA_ARGS="$W --cameras head_left,wrist_right,wrist_left" ...

# E9  larger batch -- you are at 6-19% MFU, 20.5/40GB used.
# No script edit needed: EXTRA_ARGS is appended AFTER the hardcoded
# --batch_size 64, and argparse takes the last occurrence (verified).
EXPT_NAME=e9_batch96 EXTRA_ARGS="$W --batch_size 96" ...
```

Expected, from what has been measured:

| flag | expected | basis |
|---|---|---|
| `--tactile_mode input` | **~25-40% faster** | removes one of two full transformer passes |
| `--explicit_flash_attn` | memory down, speed modest | attention is ~5% of a layer; FFN is ~72% |
| `--image_crop 192` | 196 -> 144 tokens | attention cost ~0.55x |
| `--cameras` (3 of 4) | ~8% of forward | one fewer backbone pass |
| batch 96 | MFU up | 0.32 GB/sample measured |

---

## 5. Combined

```bash
# E10  everything that won, together
EXPT_NAME=e10_combined \
EXTRA_ARGS="$W --predict_deltas --temporal_weight_mode horizon \
            --tactile_mode input --explicit_flash_attn --image_crop 192" \
phd run -ng 4 -p shr_gpu -GR H100 -l %J.log sh ori/ori/run_ori_job.sh 4
```

---

## 6. Reading the results

Compare **only** on `val_l1/all_dims`. Train loss is NOT comparable across these:

- `--loss_dim_weight_mode file` changes the per-dim weights, so the loss scale moves
- `--predict_deltas` changes the target entirely (residual vs absolute)
- `--temporal_weight_mode` changes the per-timestep weights
- `--action_dims_mode active` drops 20 dims from the loss, which flatters it
- `--tactile_mode input` removes the `l1_tac` term

`val_l1/all_dims` is unweighted, in radians, over all 65 reconstructed columns —
`origami_validate` rebuilds the full 65 exactly as the server does, so a 45-dim
model is measured on what the robot would actually execute.

---

## 7. Deploying a winner

Every flag that changes shapes or semantics is recorded in `training_configs.json`
and read back by `vitac_policy_server.py`, so a run deploys the way it trained.
Before building the image:

```bash
cd $HOME/other/ori/ori/orivizkar-submission-v000-gpu
./scripts/sync_vendor.sh          # the image ships a COPY of the model source
./scripts/contract_local.sh       # --check's the vendor sync, then validates
```

Inference-only overrides (do not need to match training):

```
VITAC_EXPLICIT_FLASH_ATTN=1   fused attention at deploy regardless of the run
VITAC_SMOOTHING=auto          chunk-seam smoothing (default)
VITAC_OPTIMIZATION=none       skip torch.compile
```

---

## 8. Priority if GPU time is limited

1. **E0** — the gate. Without it nothing below is interpretable.
2. **E2** (deltas) — the only change targeting *why* the model is at parity with holding still.
3. **E5** (`tactile_mode input`) — biggest measured compute win, and the aux task it removes is worth 3-5%.
4. **E1** (weights) — cheap, evidence-backed.
5. Everything else.

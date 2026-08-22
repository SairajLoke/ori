# Copy-paste commands (RHEL server)

`R` is just to keep the lines short — everything else is literal.

```bash
export R=$HOME/other/ori/ori/vitacformer++
export DATASET_SRC=$HOME/other/new_data/<your_224_shortgop_root>
```

`run_ori_job.sh` `cd`s to `vitacformer++` itself, so it can be launched from
anywhere; the full path below is unambiguous. If your `phd run` invocations use a
different relative path, keep yours — only the env-var prefix matters.

---

## A. One-time prep (2 commands + 1 fix)

### A1. Regenerate the action weights ON YOUR DATA — do this first

```bash
cd $R && python tools/compute_action_weights.py \
  --dataset_root $DATASET_SRC --out action_weights.json
```

The committed file came from 14 episodes of one season on a laptop. Check the
output prints:

- fingers ordered **thumb > index > middle >> ring ≈ pinky**
- `constant dims: {58: ..., 59: ..., 64: -0.8727}` (64 is forced)
- `delta spread floored on N/65 dims` with N small (7 locally)

### A2. Does fused attention actually fire?

```bash
cd $R && python tools/check_sdpa.py --batch 64 --seq 200 --dtype bfloat16
cd $R && python tools/check_sdpa.py --batch 96 --seq 200 --dtype bfloat16
```

### A3. CRLF fix (see bottom section)

```bash
cd $HOME/other/ori/ori && sed -i 's/\r$//' \
  vitacformer++/*.sh vitacformer++/tools/*.sh orivizkar-submission-v000-gpu/scripts/*.sh
```

---

## B. Training runs

Common prefix used below (edit GPUs / precision / episodes to taste):

```bash
export BASE="USE_SCRATCH=1 MIXED_PRECISION=bf16 USE_NORMALIZATION=1 MAX_EPISODES=0"
export W="--loss_dim_weight_mode file --action_weights_json action_weights.json"
export PHD="phd run -ng 4 -p shr_gpu -GR H100 -l %J.log sh $R/run_ori_job.sh 4"
```

### E0 — baseline (the gate)

```bash
env $BASE EXPT_NAME=e0_baseline $PHD
```

### E1 — torque-derived per-dim weights

```bash
env $BASE EXPT_NAME=e1_weighted EXTRA_ARGS="$W" $PHD
```

### E2 — relative actions (predict pose delta)

```bash
env $BASE EXPT_NAME=e2_deltas EXTRA_ARGS="$W --predict_deltas" $PHD
```

### E3 — deltas + temporal weighting

```bash
env $BASE EXPT_NAME=e3_deltas_temporal \
  EXTRA_ARGS="$W --predict_deltas --temporal_weight_mode horizon --action_horizon_for_weights 25" $PHD
```

### E4 — 45-dim output (drops ring/pinky ACTUATION, not just loss)

```bash
env $BASE EXPT_NAME=e4_active45 EXTRA_ARGS="$W --action_dims_mode active" $PHD
```

### E5 — tactile as plain input: ONE transformer pass

```bash
env $BASE EXPT_NAME=e5_tactile_input EXTRA_ARGS="$W --tactile_mode input" $PHD
```

### E6 — fused attention

```bash
env $BASE EXPT_NAME=e6_flash EXTRA_ARGS="$W --explicit_flash_attn" $PHD
```

### E7 — centre-crop 224 → 192 (49 → 36 tokens/cam)

```bash
env $BASE EXPT_NAME=e7_crop192 EXTRA_ARGS="$W --image_crop 192" $PHD
```

### E8 — drop one head camera

```bash
env $BASE EXPT_NAME=e8_cam3 EXTRA_ARGS="$W --cameras head_left,wrist_right,wrist_left" $PHD
```

### E9 — batch 96 (overrides the hardcoded 64; argparse takes the last flag)

```bash
env $BASE EXPT_NAME=e9_batch96 EXTRA_ARGS="$W --batch_size 96" $PHD
```

### E10 — combined

```bash
env $BASE EXPT_NAME=e10_combined \
  EXTRA_ARGS="$W --predict_deltas --temporal_weight_mode horizon --tactile_mode input --explicit_flash_attn --image_crop 192" $PHD
```

---

## C. Short compute-only runs

E5–E9 are about speed, not quality. Cap them so they cost minutes:

```bash
env $BASE EXPT_NAME=e5_speed \
  EXTRA_ARGS="$W --tactile_mode input --num_epochs 2 --max_train_steps 60 --max_val_steps 5" $PHD
```

Then read `s/step` from `dataloader_timing.log` in the run dir, not val loss.

---

## D. Reading results

```bash
grep -h "VAL epoch" ckpt_dir/fold_plane/<run>/ori_debug_rank0.log | tail -5
```

Compare on **`all_dims`** only (radians, unweighted, all 65 reconstructed
columns). Reference: **0.044 = copy current pose**, 0.103 = dataset mean.
Train loss is NOT comparable across E1–E5 — each changes the loss definition.

---

## E. Deploy a winner

```bash
cd $HOME/other/ori/ori/orivizkar-submission-v000-gpu
./scripts/sync_vendor.sh        # image ships a COPY of the model source
./scripts/contract_local.sh     # checks the sync, builds, validates
```

---

## F. The CRLF problem

Symptom on RHEL:

```
/bin/bash^M: bad interpreter: No such file or directory
```

The `\r` becomes part of the interpreter path. `core.autocrlf=input` (already set)
only normalises on **commit** — it does not stop a CRLF file being checked out or
written by a Windows editor.

**Permanent fix** — `.gitattributes` is now committed at the repo root with
`* text=auto eol=lf`. It overrides `core.autocrlf` and applies to everyone who
clones. Renormalise the existing tree once:

```bash
cd $HOME/other/ori/ori
git add --renormalize .
git status            # review, then commit
```

**Immediate fix** for a file that is already broken:

```bash
sed -i 's/\r$//' path/to/script.sh      # no extra package needed
dos2unix path/to/script.sh              # if dos2unix is installed
```

**Check before running:**

```bash
file *.sh | grep CRLF                   # silence = clean
```

**Run a CRLF script without editing it** (one-off escape hatch):

```bash
bash <(tr -d '\r' < run_ori_job.sh) 4
```

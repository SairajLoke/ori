#!/usr/bin/env bash
# For each flag combination that changes the INFERENCE path, build an untrained
# checkpoint set via the real training entrypoint, drop it in checkpoints/<name>/,
# load it through vitac_policy_server exactly as the container would, and run
# infer(). Deletes each 376MB checkpoint after testing so the disk survives.
#
# Training-only flags (--loss_dim_weight_mode, --temporal_weight_mode) are NOT
# here: they change the loss, not what the server does.
set -uo pipefail

VF=/home/sai/Desktop/ORI/ori/vitacformer++
SUB=/home/sai/Desktop/ORI/ori/orivizkar-submission-v000-gpu
VENV=${VENV:-/home/sai/Desktop/ORI/ori/origami-inference-kit-participant/sharpa_north_ces_lite_sdk-main/.venv}
export DATASET_ROOT=${DATASET_ROOT:-/media/sai/CRUZER_BLA/ori/dataset/season_POC22061_2026_07_09_16_23_46_train/lerobot3.0_shortgop15_224}
export ORI_VIDEO_BACKEND=pyav MAX_EPISODES=3 VAL_EPISODES=0
J="--action_weights_json action_weights.json"

mk_and_test () {
  name=$1; shift
  rm -rf "$SUB/checkpoints/$name"
  ( cd "$VF" && PYTHONPATH="$VF:$VF/detr" "$VENV/bin/python3" origami_imitate_episodes.py \
      --ckpt_dir "$SUB/checkpoints/$name" --ckpt_save_epochs 99 --expt_name "$name" \
      --policy_class ACT --task_name fold_plane --batch_size 4 --seed 0 --num_epochs 1 \
      --lr 3e-4 --kl_weight 10 --hidden_dim 512 --dim_feedforward 3200 --use_tactile \
      --state_dim 65 --tb_log_freq 5 --save_untrained "$@" ) >/tmp/mk_$name.log 2>&1
  if [ $? -ne 0 ]; then echo "$name: BUILD FAILED"; grep -A3 Traceback /tmp/mk_$name.log | tail -4; return; fi

  # flatten <name>/<timestamped run>/ -> <name>/ so CKPT_DIR=<name> works in the Dockerfile
  RUN=$(ls -dt "$SUB/checkpoints/$name"/*/ 2>/dev/null | head -1)
  [ -n "$RUN" ] && mv "$RUN"* "$SUB/checkpoints/$name/" 2>/dev/null && rmdir "$RUN" 2>/dev/null

  CK=$(ls -t "$SUB/checkpoints/$name"/*.ckpt 2>/dev/null | head -1)
  [ -z "$CK" ] && { echo "$name: no .ckpt produced"; return; }

  ( cd "$SUB" && PYTHONPATH="$SUB:$SUB/vitacformer:$SUB/vitacformer/detr" \
    VITAC_CKPT_PATH="$CK" VITAC_SMOOTHING=none \
    "$VENV/bin/python3" - "$name" "$SUB/checkpoints/$name" <<'PY' 2>&1 | grep -vE "UserWarning|warnings.warn"
import sys, os, json, numpy as np, logging
logging.disable(logging.INFO)
name, D = sys.argv[1], sys.argv[2]
tc = json.load(open(os.path.join(D, "training_configs.json")))
pc = tc["policy_config"]
import vitac_policy_server as S
pol = S.TeamPolicy(action_horizon=25)
obs = {k: np.zeros(v, dtype=np.uint8) for k, v in S.REQUIRED_IMAGE_SPECS.items()}
st = np.random.RandomState(0).uniform(-1, 1, 65).astype(np.float32)
obs.update({"observation/state": st, "observation/state/joint_torque": np.zeros(65, np.float32),
            "observation/tactile": np.zeros(60, np.float32), "prompt": "fold"})
a = pol.infer(obs)
cd  = tc.get("constant_action_dims") or {}
pad = tc.get("predicted_action_dims")
held = [] if pad is None else [i for i in range(65) if i not in pad and str(i) not in cd]
ok = (a.shape == (25, 65) and a.dtype == np.float32 and np.isfinite(a).all()
      and a.flags["C_CONTIGUOUS"])
ok_c = all(abs(a[:, int(k)] - float(v)).max() < 1e-4 for k, v in cd.items()) if cd else True
ok_h = all(abs(a[:, i] - st[i]).max() < 1e-4 for i in held) if held else True
print(f"{name:16s} act_dim={pc.get('action_dim',65):3d} cams={len(pc['camera_names'])} "
      f"tac={pc.get('tactile_mode','predict'):7s} crop={str(tc.get('image_crop')):4s} "
      f"delta={str(tc.get('predict_deltas'))[:1]} flash={str(pc.get('explicit_flash_attn'))[:1]} "
      f"| reply {'OK ' if ok else 'BAD'} const {'OK ' if ok_c else 'BAD'} held {'OK ' if ok_h else 'BAD'}")
PY
  )
  find "$SUB/checkpoints/$name" -name "*.ckpt" -delete   # keep the JSONs, drop 376MB
}

echo "name             action_dim cams tactile crop delta flash | inference checks"
echo "---------------------------------------------------------------------------"
mk_and_test t_baseline
mk_and_test t_deltas       $J --predict_deltas
mk_and_test t_constants    $J --use_constant_dims
mk_and_test t_active45     $J --action_dims_mode active --use_constant_dims
mk_and_test t_tactile_in   $J --tactile_mode input
mk_and_test t_crop192      $J --image_crop 192
mk_and_test t_cam3         $J --cameras head_left,wrist_right,wrist_left
mk_and_test t_flash        $J --explicit_flash_attn
mk_and_test t_combined     $J --predict_deltas --use_constant_dims --tactile_mode input \
                              --explicit_flash_attn --image_crop 192 --loss_dim_weight_mode file
echo "---------------------------------------------------------------------------"
echo "DEPLOY PATHS DONE"

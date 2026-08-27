
import os
from pathlib import Path


# ---
TOLERANCE = 0.001 #NOTE
IS_ORIGAMI_TASK = True 
FPS = 30.0 
CHUNK_SIZE = 100 
# Camera resize target (H, W). 224x224 matches robot_io_spec.md: the organizer
# squashes native 1920x1536 straight to 224x224 with no aspect-ratio preservation
# and no letterbox. Our source videos are 480x480 (already square), so this
# reproduces the deployment transform exactly instead of the old 224x320, which
# the deploy path can never produce.
IMAGE_HW = (224, 224)
# ---

#----------- :NOTE:------------------ expts 
MASK_FINGERS = False 
MAXDURATION_IN_EPISODE_SEC =  None #120
# Dataset path comes from the DATASET_ROOT env var, which run_ori_job.sh points
# either at local scratch NVMe or at the NFS source. Fail loudly if it is unset:
# Path(None) used to raise an opaque TypeError here.
_DATASET_ROOT = os.environ.get('DATASET_ROOT')
if not _DATASET_ROOT:
    raise RuntimeError(
        "DATASET_ROOT env var is not set. Export it before launching, e.g.\n"
        "  export DATASET_ROOT=$HOME/other/new_data/larger_data\n"
        "run_ori_job.sh / backup_scratch_run_ori_job.sh normally do this for you."
    )
FULL_DATASET   =  Path(_DATASET_ROOT)

# Max episodes to use (0 = all episodes). Set via MAX_EPISODES env var.
MAX_EPISODES = int(os.environ.get('MAX_EPISODES', '0'))

# Episode indices held out for validation, as explicit indices so the holdout is
# identical across runs no matter what MAX_EPISODES is set to -- which is what
# makes two runs comparable. Set VAL_EPISODES="" to disable validation.
#   VAL_EPISODES=0,1      -> default
#   VAL_EPISODES=3,17,42  -> any indices you like
VAL_EPISODES = [int(x) for x in os.environ.get('VAL_EPISODES', '0,1').split(',') if x.strip()]
VAL_EVERY_N_EPOCHS = int(os.environ.get('VAL_EVERY_N_EPOCHS', '10'))

# Feature keys forced to identity (pass-through) even when normalization is ON.
# Lets you ablate one modality at a time without editing recommended_modes():
#   ORI_NORM_DISABLE_KEYS=observation.tactile          -> everything but tactile
#   ORI_NORM_DISABLE_KEYS=observation.tactile,action   -> and leave actions raw
# Note "observation.tactile" also covers observation.tactile_next, which is
# derived from the same normalized tensor in convert_batch.
NORM_DISABLE_KEYS = [k.strip() for k in os.environ.get('ORI_NORM_DISABLE_KEYS', '').split(',') if k.strip()]

# Local ResNet checkpoint for the vision backbone -- resolved at the bottom of
# this file, once BACKBONE is known. Precedence:
#   1. BACKBONE_WEIGHTS env var
#   2. assets/backbones/<BACKBONE>_imagenet.pth shipped next to this file
#   3. None -> torchvision downloads via torch hub (needs network + writable TORCH_HOME)
_BACKBONE_WEIGHTS_ENV = os.environ.get('BACKBONE_WEIGHTS') or None



# Path("/home/sr5/sairaj.loke/other/new_data/larger_data")
# Path("/home/sr5/sairaj.loke/other/data/season_POC22032_2026_05_14_19_21_01_train/lerobot3.0") #full_ori_12")
#----------- :NOTE:------------------- 

# TODO
# https://github.com/huggingface/lerobot/blob/main/examples/dataset/use_dataset_image_transforms.py to add these transforms 
#----------------------------- expts 


if IS_ORIGAMI_TASK:
    # ------------------ State / Observation configs ------------------
    EPISODE_LEN = 10_000
    CAMERA_NAMES = ['observation.images.head_left',
                    'observation.images.head_right',
                    'observation.images.wrist_right',
                    'observation.images.wrist_left']
    STATE_DIM = 65
    # BACKBONE: 'resnet18'/'resnet34'/'resnet50' (existing default, unchanged)
    # or a ViT: 'vit_b_16'/'vit_b_32'/'vit_l_16'/'vit_l_32'/'vit_h_14'
    # (torchvision.models constructors; see detr/models/backbone.py for what
    # each is dispatched to). vit_b_16 is the one with local weights fetched
    # to assets/backbones/ so far -- others will torch-hub download until you
    # fetch their weights the same way.
    BACKBONE = os.environ.get('ORI_BACKBONE', 'resnet18')
    # LR_BACKBONE > 0 fine-tunes the backbone at this LR; <= 0 freezes it
    # entirely for a ViT (train_backbone = lr_backbone > 0 in
    # detr/models/backbone.py). For ResNet this is a no-op distinction --
    # its freeze loop has been commented out since before this project, so it
    # is never truly frozen, only ever trained at this LR. For a ViT it is a
    # real freeze, and freezing is the recommended default on a small dataset
    # (86M+ params, a few hundred episodes): e.g. ORI_LR_BACKBONE=0.
    LR_BACKBONE = float(os.environ.get('ORI_LR_BACKBONE', '1e-5'))
    # ViT only: freeze everything except the LAST N transformer blocks (+ the
    # final LayerNorm) instead of the all-or-nothing frozen/unfrozen choice
    # above. Only takes effect when LR_BACKBONE > 0 (still gates whether the
    # backbone trains at all). Unset (default) = old all-or-nothing behavior.
    # ResNet ignores this entirely.
    _vit_unfrozen_env = os.environ.get('ORI_VIT_UNFROZEN_LAYERS')
    VIT_UNFROZEN_LAYERS = int(_vit_unfrozen_env) if _vit_unfrozen_env else None

    
    # ------------------ Dataset configs ------------------
    TOLERANCE = 0.001
    INFERENCE_DATASET_ROOT = Path("/home/sr5/sairaj.loke/other/data/season_POC22032_2026_05_14_19_21_01_train/lerobot3.0")

    # # /home/ubuntu/iros2026/Robotic_Origami_Challenge")
    INDIVID_SEASONS = [
        "season_POC22032_2026_05_14_19_21_01_train",
        "season_POC22032_2026_05_14_20_40_58_train",
        "season_POC22032_2026_05_14_21_08_06_train",
        "season_POC22032_2026_05_15_16_43_23_train"

    ]
    
    assert 1/FPS != 0 
    
    PROPRIOCEPTIVE_TEMPORAL_HORIZON = 6
    TACTILE_TEMPORAL_HORIZON        = 18

    # ---- observation-sampling cadence vs action/robot-control cadence ----
    # FPS (30) drives the ACTION chunk's spacing (DELTA_TIMESTAMPS["action"]
    # below) -- this should track the robot's actual control/execution rate,
    # which the organizer runs independently of how often we recompute a
    # chunk (receding-horizon control). It must NOT change just because our
    # own inference is slow.
    #
    # OBS_FPS is a SEPARATE rate for the observation history windows
    # (state, tactile). At inference, robot_io_spec.md gives one reading per
    # infer() call, spaced however often infer() actually gets called -- not
    # necessarily 30Hz, likely well below it for a model this size. Training
    # the history window at FPS=30 spacing when real calls arrive slower
    # means every deployment observation is temporally wrong: state/tactile
    # "6 steps back" was 0.2s ago in training, but N seconds ago in reality,
    # and the tactile DELTA channel especially (which encodes an implicit
    # 1/FPS time unit into its magnitude) comes out badly mis-scaled.
    #
    # USE_OBS_FPS switches the observation windows (NOT the action chunk)
    # onto OBS_FPS instead of FPS, so both variants can be trained from the
    # same config with one env var:
    #   ORI_USE_OBS_FPS=0 (default)  -> old behavior, observation windows at FPS=30
    #   ORI_USE_OBS_FPS=1            -> observation windows at OBS_FPS
    # Step COUNTS (PROPRIOCEPTIVE_TEMPORAL_HORIZON / TACTILE_TEMPORAL_HORIZON)
    # are unchanged either way -- widening the window in TIME, not in the
    # number of samples, means qpos_dim/tactile_dim and therefore the model
    # architecture are identical between the two variants; only what the
    # window covers in real seconds differs. That also means a checkpoint
    # trained with one setting is semantically incompatible with the other
    # (like the earlier tactile-ordering fix) even though shapes match.
    #
    # Pick OBS_FPS as a divisor of FPS (30, 15, 10, 6, 5, 3, 2, 1...) so every
    # requested offset lands exactly on a native 30fps frame boundary and
    # TOLERANCE does not need loosening. A non-divisor still works but each
    # sample may be pulled from up to ~1/(2*FPS) away from the requested time.
    USE_OBS_FPS = os.environ.get('ORI_USE_OBS_FPS', '0') in ('1', 'true', 'True')
    OBS_FPS = float(os.environ.get('ORI_OBS_FPS', '5.0'))
    if USE_OBS_FPS and FPS % OBS_FPS != 0:
        print(f"[configs] WARNING: ORI_OBS_FPS={OBS_FPS} does not evenly divide "
              f"FPS={FPS} -- observation timestamps will not land exactly on "
              f"native frame boundaries. Consider a divisor of {FPS:.0f} "
              f"(30, 15, 10, 6, 5, 3, 2, 1...) or loosen TOLERANCE.")
    _OBS_SAMPLE_FPS = OBS_FPS if USE_OBS_FPS else FPS

    # ---- history jitter: emulate irregular/lagged inference-time sampling ----
    # OBS_FPS/USE_OBS_FPS above assumes deployment calls infer() at one known,
    # REGULAR rate. In reality infer() cadence is unknown and likely irregular
    # (queueing, variable model latency, etc). If training only ever sees
    # perfectly regular history spacing, the model has never seen the ragged
    # gaps deployment will actually produce.
    #
    # Two input windows are exposed to this: observation.state (past window)
    # and observation.tactile (past window only -- its delta channel, an
    # explicit torch.diff, is the more dangerous case since its magnitude
    # scales with elapsed time). NOT affected: images (single current frame,
    # no history window at all) and the two things we PREDICT -- action and
    # tactile_next -- whose spacing is our own modeling choice (robot-control
    # rate / prediction horizon), not something inference cadence imposes on
    # us, so they always stay on the regular grid.
    #
    # Mechanics: LeRobotDataset's delta_timestamps is fixed once at dataset
    # construction, so it cannot be redrawn every training step. Instead we
    # request a DENSE POOL of every native-FPS frame here (once, wide enough
    # to cover the worst-case max gap), and dataset/origami_dataset.py's
    # convert_batch draws fresh random per-step gaps -- independently per
    # sample, not just per batch -- on every call and subsamples the pool with
    # them. The fetch is fixed; the realized spacing is not. This is cheap:
    # state/tactile are small float arrays (not video), so a wider window is a
    # few extra KB, not a decode.
    JITTER_HISTORY = os.environ.get('ORI_JITTER_HISTORY', '0') in ('1', 'true', 'True')
    # Max per-step gap as a multiple of the regular (unjittered) step gap in
    # native frames. 1.0 would degenerate to the regular grid; higher = wider
    # possible gaps = exposure to slower/more irregular inference cadences.
    JITTER_MAX_GAP_MULT = float(os.environ.get('ORI_JITTER_MAX_GAP_MULT', '3.0'))
    # Regular per-step gap in native frames (e.g. FPS=30, OBS_FPS=5 -> 6).
    JITTER_BASE_GAP_FRAMES = max(1, round(FPS / _OBS_SAMPLE_FPS))
    JITTER_MAX_GAP_FRAMES = max(1, round(JITTER_BASE_GAP_FRAMES * JITTER_MAX_GAP_MULT))
    # Dense pool sizes (frame COUNTS, native-FPS-spaced, each pool ending at
    # "now"/offset 0) sized for the worst case where every one of the
    # HORIZON-1 gaps in the window hits JITTER_MAX_GAP_FRAMES.
    STATE_POOL_LEN        = (PROPRIOCEPTIVE_TEMPORAL_HORIZON - 1) * JITTER_MAX_GAP_FRAMES + 1
    TACTILE_PAST_POOL_LEN = TACTILE_TEMPORAL_HORIZON * JITTER_MAX_GAP_FRAMES + 1

    #NOTE: this corrects but also changes input semantics...can no longer resume path
    if JITTER_HISTORY:
        _state_past_offsets   = [-1 * (i / FPS) for i in range(STATE_POOL_LEN - 1, -1, -1)]
        _tactile_past_offsets = [-1 * (i / FPS) for i in range(TACTILE_PAST_POOL_LEN - 1, -1, -1)]
    else:
        _state_past_offsets   = [-1 * (i / _OBS_SAMPLE_FPS) for i in range(PROPRIOCEPTIVE_TEMPORAL_HORIZON - 1, -1, -1)]
        _tactile_past_offsets = [-1 * (i / _OBS_SAMPLE_FPS) for i in range(TACTILE_TEMPORAL_HORIZON, -1, -1)]
    _tactile_future_offsets = [(i / _OBS_SAMPLE_FPS) for i in range(1, TACTILE_TEMPORAL_HORIZON + 1)]

    TACTILE_TEMPORAL_TOTAL_TIMESTAMPS = _tactile_past_offsets + _tactile_future_offsets
    #[ -18 -17 .... 0 ] + [1 , 2.... 18]  (frame indices; real spacing is 1/_OBS_SAMPLE_FPS,
    # or dense native-FPS spacing for the past half when JITTER_HISTORY is on)

    #==================================== OBSERVATION HISTORY/FUTURES ==============================================
    DELTA_TIMESTAMPS = {

        "observation.state" : _state_past_offsets, #last [B, 6, 65] normally; [B, STATE_POOL_LEN, 65] pool when jittered

        "action":             [  i / FPS     for i in range(CHUNK_SIZE)],         #next [B, 100, 65]  -- always robot-rate, never OBS_FPS/jittered

        "observation.tactile": TACTILE_TEMPORAL_TOTAL_TIMESTAMPS,
        #last [B, 19, 3dof*20streams]  # the delta is the second input to be concatenated afterwards in batch_process
        #19 is used to get the delta, then 19th old obs is discarded in convert batch
        # (past half is a dense pool instead of exactly 19 when JITTER_HISTORY is on;
        # future half -- the tactile_next prediction target -- is always exactly 18, unjittered)
    }

    def build_delta_timestamps(image_history=False, image_history_sec=5.0,
                                image_history_pool_fps=5.0, torque_input=False,
                                camera_names=None):
        """DELTA_TIMESTAMPS extended with image-history/torque keys, driven by
        EXPLICIT arguments (CLI-arg-sourced -- origami_imitate_episodes.py/
        origami_inference.py both call this instead of reading env vars) so
        every choice made here is recorded in training_configs.json via the
        normal config dict, not silently invisible to it.

        image_history: adds a low-freq pool per camera (Gaussian-mode subsampled
        in dataset/origami_dataset.py::convert_batch -- unlike JITTER_HISTORY's
        native-FPS dense pool, images are video, so a native-FPS pool spanning
        several seconds would be 150+ decoded frames per camera per sample;
        this pool is fetched at its own low frequency instead).
        torque_input: adds observation.state.joint_torque on the same window as
        observation.state (same [65] per-frame shape, confirmed from
        meta/info.json -- identical 65 joint names).

        Returns a NEW dict each call -- callers must not mutate the module-level
        DELTA_TIMESTAMPS in place.
        """
        dt = dict(DELTA_TIMESTAMPS)
        if image_history:
            camera_names = camera_names if camera_names is not None else CAMERA_NAMES
            pool_offsets = [
                -1 * (i / image_history_pool_fps)
                for i in range(round(image_history_sec * image_history_pool_fps), -1, -1)
            ]  # e.g. 5.0s @ 5fps -> 26 timestamps, -5.0 .. 0.0, spaced 0.2s apart
            for _cam in camera_names:
                dt[_cam] = pool_offsets
        if torque_input:
            dt["observation.state.joint_torque"] = _state_past_offsets
        return dt

    #==================================== FINGER MASKING ==============================================
    #https://sharpa-robotics.github.io/sharpa-docs/#hand-control  has joint order definition 0-22
    
    HAND_MASK = [1]*5 + [1]*4 + [1]*4 + [0]*4 + [0]*5 
    print("HAND MASK" , MASK_FINGERS, len(HAND_MASK), HAND_MASK ) #????
    #==================================== FINGER MASKING ==============================================



else: 
    
    EPISODE_LEN = 10000
    CAMERA_NAMES = ['/observe/vision/head/stereo/lefteye/rgb',
                    '/observe/vision/head/stereo/righteye/rgb',
                    '/observe/vision/right_wrist/fisheye/rgb',
                    '/observe/vision/left_wrist/fisheye/rgb']
    STATE_DIM = 58
    LR_BACKBONE = 1e-5
    BACKBONE = 'resnet18'



# stats keys dict_keys([ 'observation.tactile',   'observation.images.tactile_raw', 'observation.images.tactile_deform', 
#                        'observation.state.joint_torque', 'observation.state' ])

#                       'observation.images.head_left', 'observation.images.wrist_right','observation.images.head_right',  'observation.images.wrist_left',
#                       'observation.state.tcp', 'timestamp', 'episode_index', 'action', 'index',  'frame_index', 'task_index'

# {'dtype': 'float32', 'shape': (65,), 'names': [ 'left_arm_j0', 'left_arm_j1', 'left_arm_j2', 'left_arm_j3', 'left_arm_j4', 'left_arm_j5', 'left_arm_j6', 
                                                
#                                                 'left_hand_j0', 'left_hand_j1', 'left_hand_j2', 'left_hand_j3', 'left_hand_j4', 'left_hand_j5', 'left_hand_j6', 
#                                                 'left_hand_j7', 'left_hand_j8', 'left_hand_j9', 'left_hand_j10', 'left_hand_j11', 'left_hand_j12', 'left_hand_j13', 'left_hand_j14',
#                                                 'left_hand_j15', 'left_hand_j16', 'left_hand_j17', 'left_hand_j18', 'left_hand_j19', 'left_hand_j20', 'left_hand_j21', 
                                                
#                                                 'right_arm_j0', 'right_arm_j1', 'right_arm_j2', 'right_arm_j3', 'right_arm_j4', 'right_arm_j5', 'right_arm_j6', 
                                                
#                                                 'right_hand_j0', 'right_hand_j1', 'right_hand_j2', 'right_hand_j3', 'right_hand_j4', 'right_hand_j5', 'right_hand_j6', 'right_hand_j7', 
#                                                 'right_hand_j8', 'right_hand_j9', 'right_hand_j10', 'right_hand_j11', 'right_hand_j12', 'right_hand_j13', 'right_hand_j14', 
#                                                 'right_hand_j15', 'right_hand_j16', 'right_hand_j17', 'right_hand_j18', 'right_hand_j19', 'right_hand_j20', 'right_hand_j21', 
                                                
#                                                 'motor_j0', 'motor_j1', 'motor_j2', 'motor_j3', 'motor_j4', 'motor_j5', 'motor_j6']}

# ---------------------------------------------------------------------------
# Backbone weights resolution (needs BACKBONE, defined in the branches above)
# ---------------------------------------------------------------------------
# Keeping the weights in-repo makes a run self-contained: no torch hub download,
# no dependency on a writable TORCH_HOME, and identical weights on every node.
_LOCAL_BACKBONE = Path(__file__).resolve().parent / "assets" / "backbones" / f"{BACKBONE}_imagenet.pth"
if _BACKBONE_WEIGHTS_ENV:
    BACKBONE_WEIGHTS = _BACKBONE_WEIGHTS_ENV
elif _LOCAL_BACKBONE.exists():
    BACKBONE_WEIGHTS = str(_LOCAL_BACKBONE)
else:
    BACKBONE_WEIGHTS = None

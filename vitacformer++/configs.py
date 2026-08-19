
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
    
    #NOTE: this corrects but also changes input semantics...can no longer resume path 
    TACTILE_TEMPORAL_TOTAL_TIMESTAMPS  = [ -1*(i/FPS)  for i in range(TACTILE_TEMPORAL_HORIZON, -1, -1)] \
                                       + [    (i/FPS)  for i in range(1, TACTILE_TEMPORAL_HORIZON +1)] 
    #[ -18 -17 .... 0 ] + [1 , 2.... 18]
    
    

    #==================================== OBSERVATION HISTORY/FUTURES ==============================================
    DELTA_TIMESTAMPS = {

        "observation.state" : [ -1* (i/ FPS) for i in range(PROPRIOCEPTIVE_TEMPORAL_HORIZON-1,-1,-1)], #last [B, 6, 65]

        "action":             [  i / FPS     for i in range(CHUNK_SIZE)],         #next [B, 100, 65]
        
        "observation.tactile": TACTILE_TEMPORAL_TOTAL_TIMESTAMPS, 
        #last [B, 19, 3dof*20streams]  # the delta is the second input to be concatenated afterwards in batch_process
        #19 is used to get the delta, then 19th old obs is discarded in convert batch 
    }

    
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

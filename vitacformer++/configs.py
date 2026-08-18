
import os
from pathlib import Path


# ---
TOLERANCE = 0.001 #NOTE
IS_ORIGAMI_TASK = True 
FPS = 30.0 
CHUNK_SIZE = 100 
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

# Episodes held out for validation. These are taken from the END of the loaded
# episode range so that changing MAX_EPISODES keeps the train set a prefix.
NUM_VAL_EPISODES = int(os.environ.get('NUM_VAL_EPISODES', '2'))
VAL_EVERY_N_EPOCHS = int(os.environ.get('VAL_EVERY_N_EPOCHS', '10'))



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
    LR_BACKBONE = 1e-5
    BACKBONE = 'resnet18'

    
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
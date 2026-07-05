import Path 

TOLERANCE = 0.001
IS_ORIGAMI_TASK = True 


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
    DATASET_ROOT = Path("/home/ubuntu/iros2026/Robotic_Origami_Challenge")

    SEASONS = [
        "season_POC22032_2026_05_14_19_21_01_train",
        "season_POC22032_2026_05_14_20_40_58_train",
        # Add your new seasons here as you download them:
        # "season_POC22033_train", 
        # "season_POC22034_train",
    ]
    #TODO : check what they had 
    # delta_timestamps = {
    #     "observation.images.head_left": [-0.2, -0.1, 0.0]
    # }
    
    
    # ------------------ Training configs ------------------
    BATCH_SIZE = 8
    
    
    
    
else: 
    
    EPISODE_LEN = 10000
    CAMERA_NAMES = ['/observe/vision/head/stereo/lefteye/rgb',
                    '/observe/vision/head/stereo/righteye/rgb',
                    '/observe/vision/right_wrist/fisheye/rgb',
                    '/observe/vision/left_wrist/fisheye/rgb']
    STATE_DIM = 58
    LR_BACKBONE = 1e-5
    BACKBONE = 'resnet18'

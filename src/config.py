from collections import defaultdict
from dynaconf import Dynaconf
from pathlib import Path
import numpy as np

# Dynaconf settings
SETTINGS = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=["settings.toml", ".secrets.toml"],
)

# General Preprocessing and Data Configuration
class DataConfig:
    """General configuration for data processing and labels."""
    VIDEO_FILE_EXTENSION = ".mp4"
    FRAME_STEP_INTERVAL = 10
    EXTRACTION_FPS = 1
    TRAIN_SPLIT_RATIO = 0.7
    RANDOM_SEED = 42
    FPS = 30
    FRAME_WIDTH = 2304
    FRAME_HEIGHT = 1296
    VIDEO_BATCH_SIZE = 16
    VALID_EXTENSIONS = [".jpg", ".PNG"]
    CUT_VIDEO = ['quantex_at_home_id255237_2022_05_08_04']
    CUT_VIDEO_OFFSET = 7123 #add offset to frame number for this video
    SHIFTED_VIDEOS_OFFSETS = {'quantex_at_home_id257511_2021_07_13_01': (24, -6), # annotations are shifted by six frames from frame 30 on
                              'quantex_at_home_id257573_2021_04_02_01': (28, -2), #annotations are shifted by two frames from frame 28 on
                              'quantex_at_home_id257578_2021_05_12_04': (28, -2), #annotations are shifted by two frames from frame 28 on
                              'quantex_at_home_id257578_2021_05_12_05': (28, -2),
                              'quantex_at_home_id257578_2021_05_12_06': (28, -2),
                              'quantex_at_home_id257578_2021_05_16_01': (28, -2)} #annotations are shifted by two frames from frame 28 on
    NON_STANDARD_FRAME_STEPS = {'quantex_at_home_id258704_2022_05_15_01': 34} # Interval is 34 frames, not 30
    

class LabelMapping:
    """Mappings for labels, IDs, and supercategories."""
    LABEL_TO_ID_MAPPING = defaultdict(
        lambda: 99,
        {
            "person": 1,
            "reflection": 2,
            "book": 3,
            "animal": 4,
            "toy": 5,
            "kitchenware": 6,
            "screen": 7,
            "food": 8,
            "object": 9,
            "other_object": 12,
            "face": 10,
            'child_body_parts': 11,
            "voice": 20,
            "noise": -1,
        },
    )

    ID_TO_SUPERCATEGORY_MAPPING = defaultdict(
        lambda: "unknown",
        {
            1: "person",
            2: "reflection",
            3: "object",
            4: "object",
            5: "object",
            6: "object",
            7: "object",
            8: "object",
            9: "object",
            12: "object",
            10: "face",
            20: "voice",
            -1: "noise",
            99: "unknown",
        },
    )
    unknown_label_id = -1
    unknown_supercategory = "unknown"

class BookConfig:
    """Configuration for book detection."""
    MODEL_NAME = "yolo26x"
    DATABASE_CATEGORY_IDS = [73]  # COCO category ID for 'book'
    MODEL_ID = 5
    
# Specific Task Configurations
class PersonConfig:
    """Configuration for person detection and classification."""
    MODEL_SIZE = 'x'
    MODEL_NAME = f"yolo26{MODEL_SIZE}"
    # Ratio of training data to use for training
    TRAIN_SPLIT_RATIO = 0.6
    # Ratio of class-to-class samples in each dataset split
    MAX_CLASS_RATIO_THRESHOLD = 0.9
    AGE_GROUP_TO_CLASS_ID_PERSON_ONLY = {
        'Inf': 0,
        'Child': 0,
        'Teen': 0,
        'Adult': 0,
        'Dk': 0,
    }
    AGE_GROUP_TO_CLASS_ID_AGE_BINARY = {
        'Inf': 0,
        'Child': 0,
        'Teen': 1,
        'Adult': 1,
    }
    MODEL_CLASS_ID_TO_LABEL_AGE_BINARY = {
        0: "child",
        1: "adult"
    }
    MODEL_CLASS_ID_TO_LABEL_PERSON_ONLY = {0: "person"}
    DATABASE_CATEGORY_IDS = [1, 2]
    TARGET_LABELS_AGE_BINARY = ['child', 'adult']
    TARGET_LABELS_PERSON_ONLY = ['person']
    TARGET_LABELS_CLS = ['0_no_person', '1_person']
    NEGATIVE_SAMPLING_RATIO = 1 #100% negative samples compared to positive samples in train and val splits

    MIN_IDS_PER_SPLIT = 2

    NUM_EPOCHS = 300
    BATCH_SIZE = 64
    IMG_SIZE = 640
    PATIENCE = 25
    NUM_WORKERS = 24
    DEGREES = 15

    # number of videos per batch
    LR = 1e-3
    FREEZE_CNN = True
    SEQUENCE_LENGTH = 60
    DROPOUT = 0.5
    WEIGHT_DECAY = 1e-5
    FEAT_DIM = 512
    RNN_HIDDEN = 256
    RNN_LAYERS = 2
    BIDIRECTIONAL = True
    NUM_OUTPUTS = 1
    BACKBONE = 'efficientnet_b0'
    BATCH_SIZE_INFERENCE = 64
    WINDOW_SIZE = 6 # frames (6 extracted frames = 60 original frames = 2 seconds)
    STRIDE = 1 # frames (every 1st extracted frame = every 10 original frames)
    CHILD_POS_WEIGHT = 3.08  
    ADULT_POS_WEIGHT = 2.60
    MODEL_ID = 2
    CONFIDENCE_THRESHOLD = 0.170
    
class FaceConfig:
    """Configuration for face detection and classification."""
    MODEL_SIZE = 'l'
    MODEL_NAME = f"yolo12{MODEL_SIZE}"
    AGE_GROUP_TO_CLASS_ID_AGE_BINARY = {
        'infant': 0,
        'child': 0,
        'teen': 1,
        'adult': 1,
    }
    AGE_GROUP_TO_CLASS_ID_FACE_ONLY = {
        'infant': 0,
        'child': 0,
        'teen': 0,
        'adult': 0,
    }
    MODEL_CLASS_ID_TO_LABEL_AGE_BINARY = {
        0: "child",
        1: "adult"
    }
    MODEL_CLASS_ID_TO_LABEL_FACE_ONLY = {0: "face"}
    DATABASE_CATEGORY_IDS = [10]
    TARGET_LABELS_AGE_BINARY = ['child', 'adult']
    TARGET_LABELS_FACE_ONLY = ['face']
    NEGATIVE_SAMPLING_RATIO = 1 #100% negative samples compared to positive samples in train and val splits

    TRAIN_SPLIT_RATIO = 0.6
    MIN_IDS_PER_SPLIT = 2
    
    NUM_EPOCHS = 300
    BATCH_SIZE = 64
    IMG_SIZE = 640
    PATIENCE = 30
    MODEL_ID = 1
    NUM_WORKERS = 12
    DEGREES = 15
    
    CLUSTER_CONSECUTIVE_FRAMES = 1
    REPRESENTATIVE_BLUR_THRESHOLD = 60
    MIN_BLUR_THRESHOLD = 20
    DEEPFACE_BACKEND = "retinaface"
    FINAL_CONFIRMATION_DISTANCE_THRESHOLD = 0.6
    VERIFIED_DISTANCE_THRESHOLD = 0.68
    CONFIDENCE_THRESHOLD = 0.55

class AudioConfig:
    """Configuration for audio classification."""
    VALID_RTTM_CLASSES = ['OHS', 'KCDS', 'KCHI']
    VALID_EVENT_IDS = {"child_talking", "other_person_talking", "overheard_speech", "singing/humming"}
    SR = 16000
    N_MELS = 256
    HOP_LENGTH = 256
    #WINDOW_DURATION = 3.0
    #WINDOW_STEP = 1.0
    MAX_SEGMENT_DURATION = 20 #in seconds
    FIXED_TIME_STEPS = int(np.ceil(MAX_SEGMENT_DURATION * SR / HOP_LENGTH))

    NUM_EPOCHS = 200
    PATIENCE = 30
    MODEL_ID = 3

class KchiVoc_Config:
    """Configuration for KCHI vocalizations."""
    MODEL_ID = 4
    
class AnalysisConfig:
    """Consolidated configuration for active social analysis rules."""
    
    # -- General --
    SOCIAL_STATE_MAPPING_BINARY = {1: 'Interacting', 2: 'Not Interacting'} 
    SOCIAL_STATE_MAPPING_TERTIARY = {1: 'Interacting', 2: 'Available', 3: 'Alone'}

    NUM_FOLDS = 5
    EXCLUSION_SECONDS = 30           
    SAMPLE_RATE = 10                 
    
    # -- Turn-Taking --
    MAX_TURN_TAKING_GAP_SEC = 8      
    MAX_SAME_SPEAKER_GAP_SEC = 2.25     

    # -- Proximity --
    PROXIMITY_THRESHOLD = 0.65    
    
    # Presence threshold  
    INSTANT_CONFIDENCE_THRESHOLD = 0.4 

    # -- KCDS (Adult Speech) --
    SUSTAINED_KCDS_WINDOW_SEC = 6.5    
    SUSTAINED_KCDS_THRESHOLD = 0.8    
    
    # -- Rule 4: Visual Persistence --
    HIGH_CONFIDENCE_PROXIMITY_THRESHOLD = 0.5
    HIGH_CONFIDENCE_FACE_CONFIDENCE = 0.6
    VISUAL_PERSISTENCE_SEC = 3.0 # long memory duration in seconds
    SHORT_TERM_VISUAL_MEMORY_SEC = 2.0  # Short memory (Flicker)

    # -- Audio-Visual Gating --
    AUDIO_VISUAL_GATING_FLOOR = 0.08 
    MIN_PRESENCE_OHS_FRACTION = 0.025 

    # -- CPD & Post-Processing --
    CPD_PENALTY = 2.8                  
    CPD_INTERACTING_THRESHOLD = 0.45 
    CPD_INTERACTING_THRESHOLD_LOW = 0.25
    CPD_TOTAL_PRESENCE_FLOOR = 0.45  

    SAME_SEGMENT_MERGE_THRESHOLD = 1   

    GAP_DEFAULT_LABEL_BINARY = "Not Interacting"
    GAP_DEFAULT_LABEL_TERTIARY = "Alone"
    GAP_STRETCH_THRESHOLD = 1   
      
class HyperparameterConfig:
    """
    Complete Search Grid for Social Interaction Analysis.
    Includes all 15 active parameters from the frame-level and video-level engines.
    """
    HYPERPARAMETER_RANGES = {
        # --- 1. Audio-Visual Gating & Confidence ---
        # Shifted Higher: We need more visual evidence before trusting audio.
        'AUDIO_VISUAL_GATING_FLOOR': [0.15, 0.22, 0.30], 
        'INSTANT_CONFIDENCE_THRESHOLD': [0.25, 0.35, 0.5],
        
        # --- 2. Turn-Taking (Rule 1) ---
        # Shifted Lower: 12s is too long; it's likely catching coincidental noises.
        'MAX_TURN_TAKING_GAP_SEC': [4.0, 6.0, 8.0],
        'MAX_SAME_SPEAKER_GAP_SEC': [1.5, 2.25, 3.0],
        
        # --- 3. Proximity (Rule 2) ---
        'PROXIMITY_THRESHOLD': [0.60, 0.70, 0.80],
        
        # --- 4. Sustained Adult Speech (Rule 3) ---
        'SUSTAINED_KCDS_WINDOW_SEC': [6.0, 8.0, 10.0],
        'SUSTAINED_KCDS_THRESHOLD': [0.75, 0.85, 0.95], # Stricter 'fill' requirement
        
        # --- 5. Visual Persistence & OHS ---
        'HIGH_CONFIDENCE_PROXIMITY_THRESHOLD': [0.5, 0.6, 0.7],
        'HIGH_CONFIDENCE_FACE_CONFIDENCE': [0.6, 0.7, 0.8],
        # Shifted Lower: 6s 'memory' is causing the Alone -> Interacting bleed.
        'VISUAL_PERSISTENCE_SEC': [1.5, 3.0, 4.5],
        'SHORT_TERM_VISUAL_MEMORY_SEC': [1.0, 2.0, 3.0], 
        'MIN_PRESENCE_OHS_FRACTION': [0.02, 0.04, 0.06],

        # --- 6. CPD Smoothing (Video Level) ---
        # Shifted Lower: 4.2 was too 'stubborn'. Lower penalty allows state changes.
        'CPD_PENALTY': [1.5, 2.5, 3.5],
        # Shifted Higher: Raise the 'bar' for a segment to be called Interacting.
        'CPD_INTERACTING_THRESHOLD': [0.50, 0.60, 0.70],
        'CPD_INTERACTING_THRESHOLD_LOW': [0.05, 0.10, 0.20],
        'CPD_TOTAL_PRESENCE_FLOOR': [0.40, 0.50, 0.60],
        
        # --- 7. Timeline Post-Processing ---
        'SAME_SEGMENT_MERGE_THRESHOLD': [0.5, 1.0, 2.0],
        'GAP_STRETCH_THRESHOLD': [0.5, 1.0, 1.5],
    }
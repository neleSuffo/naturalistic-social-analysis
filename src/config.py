#from dynaconf import Dynaconf
from collections import defaultdict
from pathlib import Path
from dynaconf import Dynaconf


settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=["settings.toml", ".secrets.toml"],
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.

# Whether to generate the output video with the detection results
generate_detection_output_video = settings.get("generate_detection_output_video", True)

class DetectionParameters:
    # the video file extension
    video_file_extension = ".mp4"
    # Every frame_step-th frame is processed
    frame_step_interval = 10
    # The class of the object to detect
    yolo_detection_target = "person"
    mtcnn_detection_target = "face"
    vtc_detection_target = "voice"
    output_key_videos = "videos"
    output_key_annotations = "annotations"
    output_key_images = "images"
    output_key_categories = "categories"
    
class YoloConfig:
    extraction_fps = 1  # the frames per second to extract from the video
    batch_size = 64
    num_epochs = 100
    iou_threshold = 0.35 # the intersection over union threshold
    img_size = (320, 640) # multi scale training
    person_target_class_ids = [1,2,10,11]
    person_object_target_class_ids = [1,2,3,4,5,6,7,8,10,11,12]
    face_target_class_ids = [10]
    detection_mapping = {
        0: 'infant/child',
        1: 'adult',
        2: 'infant/child face',
        3: 'adult face',
        4: 'child body parts',
        5: 'book',
        6: 'toy',
        7: 'kitchenware',
        8: 'screen',
        9: 'food',
        10: 'other_object'
    }
    best_iou = 0.5
    
class VTCConfig:
    audio_file_suffix = Path("_16kHz.wav")

class StrongSortConfig:
    image_subdir = "img1"
    feature_subdir = "features"
    detection_file_path = "det/det.txt"
    
class FastReIDConfig:
    trt_batch_size = 8
    trt_height = 256
    trt_width = 128
    
class ResNetConfig:
    num_epochs = 10
    batch_size = 32
    
class MtcnnConfig:
    class_ids = [1,2]

class TrainingConfig:
    train_test_split_ratio = 0.8
    random_seed = 42
    
class VideoConfig:
    fps = 30
    frame_width = 2304
    frame_height = 1296
    video_batch_size = 16

class LabelToCategoryMapping:
    # Map labels to integers (defaultdict is used to return 99 for unknown labels)
    label_to_id_mapping = defaultdict(
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

    # Map label id to supercategory
    id_to_supercategory_mapping = defaultdict(
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
    # List of labels to include in the ActivityNet format
    childlens_activities_to_include = ['Child Talking',
                                      'Other Person Talking',
                                      'Overheard Speech',
                                      'Singing/Humming',
                                      ]
    unknown_label_id = -1
    unknown_supercategory = "unknown"

class DetectionPipelineConfig:
    videos_to_not_process = [
        "quantex_at_home_id260694_2022_06_29_01.MP4",
        "quantex_at_home_id260694_2022_06_29_02.MP4",
        "quantex_at_home_id260694_2022_06_29_03.MP4",
        "quantex_at_home_id260694_2022_05_21_01.MP4",
        "quantex_at_home_id260694_2022_05_21_02.MP4",
        "quantex_at_home_id254922_2022_06_29_01.MP4",
        "quantex_at_home_id254922_2022_06_29_02.MP4",
        "quantex_at_home_id254922_2022_06_29_03.MP4",
    ]
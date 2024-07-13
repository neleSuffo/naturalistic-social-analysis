from collections import defaultdict
from pathlib import Path


class DetectionPaths:
    person = Path("src/projects/social_interactions/outputs/yolov5/")
    face = Path("src/projects/social_interactions/outputs/mtcnn/")
    videos_input = Path("../../ProcessedData/videos/") 
    images_input = Path("../../ProcessedData/images/")
    results = Path("src/projects/social_interactions/outputs/")
    frames_output = Path("src/projects/social_interactions/outputs/frames/")
    # Path variable to the annotation xml files
    annotations_folder_path = Path("data/annotations/")
    annotations_xml_path = Path("data/annotations/annotations.xml")
    annotations_json_path = Path("data/annotations/annotations.json")
    annotations_db_path = Path("databases/annotations.db")
    # The file that is used to map the file names to the file ids
    file_name_id_dict_path = Path("data/file_name_to_id_dict/annotations.xml")


class VTCParameters:
    # path variables for the voice-type-classifier
    audio_path = Path("data/audio/")
    environment_path = Path(
        "/home/nele_pauline_suffo/.conda/envs/pyannote"
    )
    # environment_path = Path("/Users/nelesuffo/Library/Caches/pypoetry/virtualenvs/pyannote-afeazePz-py3.8/bin/python")
    execution_file_path = Path(
        "src/projects/social_interactions/scripts/language/run_vtc.py"
    )
    execution_command = Path("../voice_type_classifier/apply.sh")
    output_file_path = Path(
        "src/projects/social_interactions/outputs/vtc/audio/all.rttm"
    )
    output_path = Path("src/projects/social_interactions/outputs/vtc")


class DetectionParameters:
    """
    The parameters for the detection models.
    """

    # the video file extension
    file_extension = ".mp4"

    # Every frame_step-th frame is processed
    frame_step = 30
    # The minimum length of a social interaction
    # Based on the frame step of 30 frames
    # Adjust accordingly so that the minimum length is at least 3 second
    interaction_length = 3
    # The class of the object to detect
    yolo_detection_class = "person"
    mtcnn_detection_class = "face"
    vtc_detection_class = "voice"


class LabelToCategoryMapping:
    # Map labels to integers (defaultdict is used to return 99 for unknown labels)
    label_dict = defaultdict(
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
            "other_object": 9,
            "face": 10,
            "voice": 20,
            "noise": -1,
        },
    )

    # Map label id to supercategory
    supercategory_dict = defaultdict(
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
            10: "face",
            20: "voice",
            -1: "noise",
            99: "unknown",
        },
    )

    unknown_label_id = -1
    unknown_supercategory = "unknown"


class YoloParameters:
    fps = 1  # the frames per second to extract from the video
    model_path = Path("src/projects/social_interactions/models/yolov5/model.yaml")
    hyp_path = Path("src/projects/social_interactions/models/yolov5/hyp.yaml")
    pretrained_weights_path = Path("pretrained_models/yolov5s.pt")
    data_config_path = Path(
        "src/projects/social_interactions/models/yolov5_inference/dataset.yaml"
    )
    yolov5_repo_path = Path("../yolov5")
    labels_input = Path("../../ProcessedData/yolo_labels/")
    # the path to the input folder for the yolo model
    data_input = Path("../../ProcessedData/yolo/")
    batch_size = 16
    epochs = 100
    img_size = 640


class MtcnnParameters:
    labels_input = Path("../../ProcessedData/mtcnn/labels.txt")
    data_input = Path("../../ProcessedData/mtcnn/")


class TrainParameters:
    train_test_split = 0.8
    random_seed = 42


class VideoParameters:
    # The standard frame width and height
    frame_width = 2304
    frame_height = 1296
    # The number of videos to process concurrently
    batch_size = 16
    # Define the path for the success log file 
    # (contains the paths of the successfully processed videos)
    success_log_path = Path("src/projects/shared/process_data/output/success.log")

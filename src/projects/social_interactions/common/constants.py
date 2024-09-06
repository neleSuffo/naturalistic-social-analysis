from collections import defaultdict
from pathlib import Path



class DetectionPaths:
    home_dir = Path("/home/nele_pauline_suffo")
    leuphana_ipe_dir = Path(home_dir/"projects/leuphana_ipe")
    models_dir = Path(home_dir/"models")
    data_dir = Path(home_dir/"ProcessedData")
    
    person = Path(leuphana_ipe_dir/"outputs/yolo/")
    face = Path(leuphana_ipe_dir/"outputs/mtcnn/")
    #videos_input = Path(data_dir/"videos_train/") 
    #videos_input = Path(data_dir/"videos/") 
    videos_input = Path(data_dir/"videos_example/") 
    images_input = Path(data_dir/"images/")
    results = Path(leuphana_ipe_dir/"outputs")
    # Path variable to the annotation xml files
    annotations_folder_path = Path("../../ProcessedData/annotations/")
    annotations_xml_path = Path("../../ProcessedData/annotations/annotations.xml")
    annotations_individual_folder_path = Path("../../ProcessedData/annotations_individual/")
    annotations_json_path = Path("../../ProcessedData/annotations/annotations.json")
    annotations_db_path = Path("../../ProcessedData/databases/annotations.db")
    combined_json_output_path = results/"combined_detections.json"

    # The file that is used to map the file names to the file ids
    file_name_id_dict_path = Path("../../ProcessedData/file_name_to_id_dict/annotations.xml")

class DetectionParameters:
    """
    The parameters for the detection models.
    """

    # the video file extension
    file_extension = ".mp4"

    # Every frame_step-th frame is processed
    frame_step = 30
    # The class of the object to detect
    yolo_detection_class = "person"
    mtcnn_detection_class = "face"
    vtc_detection_class = "voice"
    output_key_videos = "videos"
    output_key_annotations = "annotations"
    output_key_images = "images"
    output_key_categories = "categories"

class YoloParameters:
    fps = 1  # the frames per second to extract from the video
    trained_weights_path = Path(DetectionPaths.models_dir/'yolov8_trained.pt')
    data_config = Path(DetectionPaths.leuphana_ipe_dir/"src/projects/social_interactions/models/yolo_inference/dataset.yaml")
    labels_input = Path(DetectionPaths.data_dir/"yolo_labels")
    # the path to the input folder for the yolo model
    data_input = Path(DetectionPaths.data_dir/"yolo")
    batch_size = 64
    epochs = 100
    iou_threshold = 0.35 # the intersection over union threshold
    img_size = (320, 640) # multi scale training
    class_id = [1,2,11]

class VTCParameters:
    # path variables for the voice-type-classifier
    audio_path = Path("../../ProcessedData/audio/")
    audio_name_ending = Path("_16kHz.wav")
    environment_path = Path( "/home/nele_pauline_suffo/.conda/envs/pyannote/bin/python")
    execution_file_path = Path("src/projects/social_interactions/scripts/language/run_vtc.py")
    execution_command = Path("../voice_type_classifier/apply.sh")
    output_file_path = Path("outputs/vtc/audio/all.rttm")
    output_path = Path("outputs/vtc")
    repo_path = Path("../voice_type_classifier")
    vtc_input_path = Path("../leuphana_ipe/ProcessedData/audio/")
    df_output_file_path = Path("outputs/vtc/audio_data.pkl")

class StrongSortParameters:
    base_folder = Path("/home/nele_pauline_suffo/ProcessedData/strong_sort/")
    video_input = Path(f"{base_folder}/quantex/")
    videos_train = Path(f"{video_input}/train/")
    videos_val = Path(f"{video_input}/val/")
    image_subfolder = "img1"
    feature_subfolder = "features"
    detection_file = "det/det.txt"
    ecc_output = Path({base_folder}/"ECC_train.json")
    
class FastReIDParameters:
    base_folder = Path("/home/nele_pauline_suffo/ProcessedData/fast_re_id/")
    pretrained_model = Path(DetectionPaths.models_dir/"duke_agw_R101-ibn.pth")
    config_file = Path(DetectionPaths.leuphana_ipe_dir/"src/projects/social_interactions/models/fast_re_id/dukemtmc_agw_resnet101_ibn.yaml")
    
class ResNetParameters:
    epochs = 10
    trained_model_path = Path('gaze_estimation_model.pth')
    gaze_label_csv = Path('../../ProcessedData/gaze_labels.csv')
    batch_size = 32
    
class MtcnnParameters:
    labels_input = Path("../../ProcessedData/mtcnn/labels.txt")
    data_input = Path("../../ProcessedData/mtcnn/")
    class_id = [1,2]

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
            'child_body_parts': 11,
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
    
    
class ModelsPreprocessing:
    yolo = "yolo"
    mtcnn = "mtcnn"
    fast_re_id = "fast_re_id"
    strong_sort = "strong_sort"
    gaze_estimation = "gaze_estimation"
    vtc = "vtc"
    openpose = "openpose"
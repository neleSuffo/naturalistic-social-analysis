from pathlib import Path

class BasePaths:
    home_dir = Path("/home/nele_pauline_suffo")
    models_dir = Path(home_dir/"models")
    data_dir = Path(home_dir/"ProcessedData")
    output_dir = Path(home_dir/"outputs")
    leuphana_ipe_dir = Path(home_dir/"projects/leuphana_ipe")
    vtc_dir = Path(home_dir/"projects/voice_type_classifier")
    strong_sort_dir = Path(home_dir/"projects/strong_sort")
    fast_re_id_dir = Path(home_dir/"projects/fast_re_id")

class DetectionPaths:    
    person_detections_dir = Path(BasePaths.leuphana_ipe_dir/"outputs/yolo/")
    face_detections_dir = Path(BasePaths.leuphana_ipe_dir/"outputs/mtcnn/")
    #videos_input_dir = Path(BasePaths.data_dir//"videos_train/") 
    #videos_input_dir = Path(BasePaths.data_dir//"videos/") 
    videos_input_dir = Path(BasePaths.data_dir/"videos_example/") 
    images_input_dir = Path(BasePaths.data_dir/"images/")
    # Path variable to the annotation xml files
    annotations_dir = Path(BasePaths.data_dir/"annotations/")
    annotations_xml_path = Path(annotations_dir/"annotations.xml")
    annotations_individual_dir = Path(BasePaths.data_dir/"annotations_individual/")
    annotations_json_path = Path(annotations_dir/"annotations.json")
    annotations_db_path = Path(BasePaths.data_dir/"databases/annotations.db")
    combined_json_output_path = Path(BasePaths.output_dir/"combined_detections.json")
    # The file that is used to map the file names to the file ids
    file_name_id_mapping_path = Path(BasePaths.data_dir/"file_name_to_id_dict/annotations.xml")

class YoloPaths:
    trained_weights_path = Path(BasePaths.models_dir/'yolov8_trained.pt')
    data_config_path = Path(BasePaths.leuphana_ipe_dir/"src/projects/social_interactions/models/yolo_inference/dataset.yaml")
    labels_input_dir = Path(BasePaths.data_dir/"yolo_labels")
    # the path to the input folder for the yolo model
    data_input_dir = Path(BasePaths.data_dir/"yolo")

class VTCPaths:
    audio_dir = Path(BasePaths.data_dir/"audio")
    python_env_path = Path(BasePaths.home_dir/".conda/envs/pyannote/bin/python")
    script_path = Path("src/projects/social_interactions/scripts/language/run_vtc.py")
    command_script_path = Path(BasePaths.vtc_dir/"apply.sh")
    output_rttm_path = Path(BasePaths.output_dir/"vtc/audio/all.rttm")
    output_dir = Path(BasePaths.output_dir/"vtc")
    df_output_pickle = Path(BasePaths.output_dir/"vtc/audio_data.pkl")

class StrongSortPaths:
    base_dir = Path(BasePaths.data_dir/"strong_sort/")
    video_input_dir = Path(base_dir/"quantex/")
    train_videos_dir = Path(video_input_dir/"train/")
    val_videos_dir = Path(video_input_dir/"val")
    ecc_output_path = Path({base_dir}/"ECC_train.json")
    
class FastReIDPaths:
    base_dir = Path(BasePaths.data_dir/"fast_re_id/")
    python_env_path = Path(BasePaths.home_dir/".cache/pypoetry/virtualenvs/fastreid-GjhRcNOO-py3.8")
    pretrained_model_path = Path(DetectionPaths.models_dir/"duke_agw_R101-ibn.pth")
    config_file_path = Path(DetectionPaths.leuphana_ipe_dir/"src/projects/social_interactions/models/fast_re_id/dukemtmc_agw_resnet101_ibn.yaml")
    trt_engine_path = Path(BasePaths.models_dir/"duke_R101.engine")
    output_dir = Path(BasePaths.output_dir/"fast_reid/")
    
class ResNetPaths:
    trained_model_path = Path('gaze_estimation_model.pth')
    gaze_labels_csv_path = Path(BasePaths.data_dir/'gaze_labels.csv')
    
class MtcnnPaths:
    data_dir = Path(BasePaths.data_dir/"mtcnn/")
    labels_file_path = Path(data_dir/"labels.txt")

class VideoParameters:
    success_log_path = Path("src/projects/shared/process_data/output/success.log")
    
class ModelNames:
    yolo_model = "yolo"
    mtcnn_model = "mtcnn"
    fast_reid_model = "fast_re_id"
    strong_sort_model = "strong_sort"
    gaze_estimation_model = "gaze_estimation"
    vtc_model = "vtc"
    openpose_model = "openpose"
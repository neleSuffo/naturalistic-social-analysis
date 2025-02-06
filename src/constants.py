from pathlib import Path

class BasePaths:
    home_dir = Path("/home/nele_pauline_suffo")
    models_dir = Path(home_dir/"models")
    data_dir = Path(home_dir/"ProcessedData")
    output_dir = Path(home_dir/"outputs")
    leuphana_ipe_dir = Path(home_dir/"projects/leuphana-IPE")
    vtc_dir = Path(home_dir/"projects/voice_type_classifier")
    strong_sort_dir = Path(home_dir/"projects/StrongSORT")
    fast_re_id_dir = Path(home_dir/"projects/fast-reid")

class DetectionPaths:    
    person_detections_dir = Path(BasePaths.home_dir/"outputs/yolov8/")
    face_detections_dir = Path(BasePaths.home_dir/"outputs/mtcnn/")
    #videos_input_dir = Path(BasePaths.data_dir//"videos_train/") 
    quantex_videos_input_dir = Path(BasePaths.data_dir/"quantex_videos/") 
    videos_input_dir = Path(BasePaths.data_dir/"videos_superannotate_all/") 
    images_input_dir = Path(BasePaths.data_dir/"quantex_videos_processed/")
    gaze_images_input_dir = Path(BasePaths.data_dir/"quantex_rawframes_gaze/")
    # Path variable to the annotation xml files
    annotations_dir = Path(BasePaths.data_dir/"quantex_annotations/")
    annotations_xml_path = Path(annotations_dir/"annotations.xml")
    annotations_individual_dir = Path(BasePaths.data_dir/"quantex_annotations_individual/")
    annotations_json_path = Path(annotations_dir/"annotations.json")
    annotations_db_path = Path(annotations_dir/"annotations.db")
    combined_json_output_path = Path(BasePaths.output_dir/"combined_detections.json")
    # The file that is used to map the file names to the file ids
    file_name_id_mapping_path = Path(BasePaths.data_dir/"quantex_file_name_to_id_dict/annotations.xml")

class YoloPaths:
    person_trained_weights_path = Path(BasePaths.models_dir/'yolov8_person_detection.pt')
    person_data_config_path = Path(BasePaths.leuphana_ipe_dir/"src/models/yolo_person_detection/person_dataset.yaml")
    person_labels_input_dir = Path(BasePaths.data_dir/"yolo_person_labels")
    person_data_input_dir = Path(BasePaths.data_dir/"yolo_person_input")
    person_output_dir = Path(BasePaths.output_dir/"yolo_person_detections/")

    face_trained_weights_path = Path(BasePaths.models_dir/'yolo11_face_detection.pt')
    face_data_config_path = Path(BasePaths.leuphana_ipe_dir/"src/models/yolo_face_detection/face_dataset.yaml")
    face_labels_input_dir = Path(BasePaths.data_dir/"yolo_face_labels")
    face_data_input_dir = Path(BasePaths.data_dir/"yolo_face_input")
    face_output_dir = Path(BasePaths.output_dir/"yolo_face_detections/")

    gaze_extraction_progress_file_path = Path(BasePaths.data_dir/"gaze_extraction_progress.txt")
    gaze_missing_frames_file_path = Path(BasePaths.data_dir/"gaze_missing_frames.txt")
    gaze_data_config_path = Path(BasePaths.leuphana_ipe_dir/"src/models/yolo_gaze_classification/gaze_dataset.yaml")
    gaze_labels_input_dir = Path(BasePaths.data_dir/"yolo_gaze_labels")
    gaze_data_input_dir = Path(BasePaths.data_dir/"yolo_gaze_input")
    gaze_output_dir = Path(BasePaths.output_dir/"yolo_gaze_classification/")

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
    python_env_path = Path(BasePaths.home_dir/".conda/envs/strongsort/bin/python")
    dataset_name = "quantex"
    video_input_dir = Path(base_dir/"quantex/")
    train_videos_dir = Path(video_input_dir/"train/")
    val_videos_dir = Path(video_input_dir/"val")
    ecc_train_output_path = Path(f"{base_dir}/{dataset_name}_ECC_train.json")
    ecc_val_output_path = Path(f"{base_dir}/{dataset_name}_ECC_val.json")
    ecc_script = Path(BasePaths.strong_sort_dir/"others/ecc.py")
    deep_sort_output_dir = Path(BasePaths.output_dir/"deep_sort/")
    strong_sort_output_dir = Path(BasePaths.output_dir/"strong_sort/")

class FastReIDPaths:
    base_dir = Path(BasePaths.data_dir/"fast_re_id/")
    python_env_path = Path(BasePaths.home_dir/".cache/pypoetry/virtualenvs/fastreid-GjhRcNOO-py3.8")
    images_val_dir = Path(base_dir/"quantex/train/")
    images_train_dir = Path(base_dir/"quantex/val/")
    pretrained_model_path = Path(BasePaths.models_dir/"duke_agw_R101-ibn.pth")
    config_file_path = Path(BasePaths.leuphana_ipe_dir/"src/projects/social_interactions/models/fast_re_id/dukemtmc_agw_resnet101_ibn.yaml")
    trt_engine_path = Path(BasePaths.models_dir/"duke_R101.engine")
    output_dir_train = Path(StrongSortPaths.base_dir/"quantex_test_YOLO+BoT/")
    output_dir_val = Path(StrongSortPaths.base_dir/"quantex_val_YOLO+BoT/")

class EfficientNetPaths:
    output_dir = Path(BasePaths.output_dir/"efficientnet/")

class MtcnnPaths:
    data_dir = Path(BasePaths.data_dir/"mtcnn/")
    labels_file_path = Path(data_dir/"face_labels.txt")
    faces_dir = Path(BasePaths.data_dir/"quantex_faces/")
    face_detection_results_file_path = Path(BasePaths.output_dir/"mtcnn/face_detection_results.txt")
    output_dir = Path(BasePaths.output_dir/"mtcnn/")
    
class VideoParameters:
    success_log_path = Path("src/projects/shared/process_data/output/success.log")
    rawframes_extraction_error_log = Path(BasePaths.output_dir/"rawframes_extraction_error.log")
    
class ModelNames:
    yolo_model = "yolo"
    mtcnn_model = "mtcnn"
    fast_reid_model = "fast_re_id"
    strong_sort_model = "strong_sort"
    gaze_estimation_model = "gaze_estimation"
    vtc_model = "vtc"
    openpose_model = "openpose"

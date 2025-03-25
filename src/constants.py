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
    childlens_videos_input_dir = Path(BasePaths.data_dir/"childlens_videos/") 
    images_input_dir = Path(BasePaths.data_dir/"quantex_videos_processed/")
    childlens_images_input_dir = Path(BasePaths.data_dir/"childlens_videos_processed/")
    gaze_images_input_dir = Path(BasePaths.data_dir/"quantex_rawframes_gaze/")
    # Path variable to the annotation xml files
    quantex_annotations_dir = Path(BasePaths.data_dir/"quantex_annotations/")
    childlens_annotations_dir = Path(BasePaths.data_dir/"childlens_annotations/")
    annotations_xml_path = Path(quantex_annotations_dir/"annotations.xml")
    annotations_individual_dir = Path(BasePaths.data_dir/"quantex_annotations_individual/")
    annotations_json_path = Path(quantex_annotations_dir/"annotations.json")
    quantex_annotations_db_path = Path(quantex_annotations_dir/"quantex_annotations.db")
    detection_results_dir = Path(BasePaths.output_dir/"detection_pipeline_results/")
    detection_db_path = Path(detection_results_dir/'detection_results.db')
    # The file that is used to map the file names to the file ids
    file_name_id_mapping_path = Path(BasePaths.data_dir/"quantex_file_name_to_id_dict/annotations.xml")

class YoloPaths:
    yolo_detections_dir = Path(BasePaths.leuphana_ipe_dir/"src/models/yolo_detections/")
    
    all_trained_weights_path = Path(BasePaths.models_dir/'yolo11_all_detection.pt')
    all_data_config_path = yolo_detections_dir/"yolo_all_dataset.yaml"
    all_labels_input_dir = Path(BasePaths.data_dir/"yolo_all_labels")
    all_data_input_dir = Path(BasePaths.data_dir/"yolo_all_input")
    all_output_dir = Path(BasePaths.output_dir/"yolo_all_detections/")
    
    object_trained_weights_path = Path(BasePaths.models_dir/'yolo11_object_detection.pt')
    object_data_config_path = yolo_detections_dir/"yolo_object_dataset.yaml"
    object_labels_input_dir = Path(BasePaths.data_dir/"yolo_object_labels")
    object_data_input_dir = Path(BasePaths.data_dir/"yolo_object_input")
    objectoutput_dir = Path(BasePaths.output_dir/"yolo_object_detections/")
    
    adult_person_face_trained_weights_path = Path(BasePaths.models_dir/'yolo11_adult_person_face_detection.pt')
    adult_person_face_data_config_path = yolo_detections_dir/"yolo_adult_person_face_dataset.yaml"
    adult_person_face_labels_input_dir = Path(BasePaths.data_dir/"yolo_adult_person_face_labels")
    adult_person_face_data_input_dir = Path(BasePaths.data_dir/"yolo_adult_person_face_input")
    adult_person_face_output_dir = Path(BasePaths.output_dir/"yolo_adult_person_face_detections/")
    
    child_person_face_trained_weights_path = Path(BasePaths.models_dir/'yolo11_child_person_face_detection.pt')
    child_person_face_data_config_path = yolo_detections_dir/"<olo_child_person_face_dataset.yaml"
    child_person_face_labels_input_dir = Path(BasePaths.data_dir/"yolo_child_person_face_labels")
    child_person_face_data_input_dir = Path(BasePaths.data_dir/"yolo_child_person_face_input")
    child_person_face_output_dir = Path(BasePaths.output_dir/"yolo_child_person_face_detections/")

    gaze_extracted_faces_dir = Path(BasePaths.data_dir/"quantex_gaze_input")
    gaze_trained_weights_path = Path(BasePaths.models_dir/'yolo11_gaze_classification.pt')
    gaze_extraction_progress_file_path = Path(BasePaths.data_dir/"gaze_extraction_progress.txt")
    gaze_missing_frames_file_path = Path(BasePaths.data_dir/"gaze_missing_frames.txt")
    gaze_data_config_path = Path(BasePaths.leuphana_ipe_dir/"src/models/yolo_gaze_classification/gaze_dataset.yaml")
    gaze_labels_input_dir = Path(BasePaths.data_dir/"yolo_gaze_labels")
    gaze_data_input_dir = Path(BasePaths.data_dir/"yolo_gaze_input_balanced")
    gaze_output_dir = Path(BasePaths.output_dir/"yolo_gaze_classification/")
    
    @classmethod
    def get_target_paths(cls, yolo_target: str, split_type: str) -> Optional[Tuple[Path, Path]]:
        """Get image and label destination paths for a given target and split type."""
        path_mapping = {
            "child_person_face": (
                cls.child_person_face_data_input_dir / "images" / split_type,
                cls.child_person_face_data_input_dir / "labels" / split_type
            ),
            "adult_person_face": (
                cls.adult_person_face_data_input_dir / "images" / split_type,
                cls.adult_person_face_data_input_dir / "labels" / split_type
            ),
            "gaze": (
                cls.gaze_data_input_dir / split_type / "gaze",
                cls.gaze_data_input_dir / split_type / "gaze"
            ),
            "no_gaze": (
                cls.gaze_data_input_dir / split_type / "no_gaze",
                cls.gaze_data_input_dir / split_type / "no_gaze"
            ),
            "object": (
                cls.object_data_input_dir / "images" / split_type,
                cls.object_data_input_dir / "labels" / split_type
            ),
            "all": (
                cls.all_data_input_dir / "images" / split_type,
                cls.all_data_input_dir / "labels" / split_type
            )
        }
        return path_mapping.get(yolo_target)

class ResNetPaths:
    trained_weights_path = Path(BasePaths.models_dir/"resnet_gaze_classification.pth")
    out_dir = Path(BasePaths.output_dir/"resnet_gaze_classification/")
    confusion_matrix_path = Path(out_dir/"confusion_matrix.png")
    
class VTCPaths:
    childlens_audio_dir = Path(BasePaths.data_dir/"childlens_audio")
    vtc_results_dir = Path("/home/nele_pauline_suffo/projects/voice_type_classifier/output_voice_type_classifier/")
    output_dir = Path(BasePaths.output_dir/"vtc")
    childlens_output_folder = Path(output_dir/"childlens_audio_duration_off_01")
    childlens_df_file_path_01 = Path(output_dir/"childlens_df_duration_off_01.pkl")
    childlens_df_file_path_02 = Path(output_dir/"childlens_df_duration_off_02.pkl")
    childlens_df_file_path_20 = Path(output_dir/"childlens_df_duration_off_20.pkl")
    childlens_gt_df_file_path = Path(DetectionPaths.childlens_annotations_dir/"childlens_annotations.pkl")

    quantex_output_folder = Path(output_dir/"quantex_audio")
    quantex_df_file_path = Path(output_dir/"quantex_df.pkl")

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
    
class Proximity:
    reference_file = Path("/home/nele_pauline_suffo/outputs/reference_proximity.json")
    child_close_image_path = Path("/home/nele_pauline_suffo/outputs/proximity_sampled_frames/child_reference_proximity_value_1.jpg")
    child_far_image_path = Path("/home/nele_pauline_suffo/outputs/proximity_sampled_frames/child_reference_proximity_value_0.jpg")
    adult_close_image_path = Path("/home/nele_pauline_suffo/outputs/proximity_sampled_frames/adult_reference_proximity_value_1.jpg")
    adult_far_image_path = Path("/home/nele_pauline_suffo/outputs/proximity_sampled_frames/adult_reference_proximity_value_0.jpg")

class DetectionPipeline:
    quantex_subjects = Path("/home/nele_pauline_suffo/ProcessedData/quantex_subjects.csv")

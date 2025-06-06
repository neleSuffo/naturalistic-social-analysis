from pathlib import Path
from typing import Optional, Tuple

VALID_TARGETS = {"person_face", "all", "person_cls", "face_cls", "gaze_cls"}

class BasePaths:
    home_dir = Path("/home/nele_pauline_suffo")
    models_dir = Path(home_dir/"models")
    data_dir = Path(home_dir/"ProcessedData")
    output_dir = Path(home_dir/"outputs")
    leuphana_ipe_dir = Path(home_dir/"projects/leuphana-IPE")
    vtc_dir = Path(home_dir/"projects/voice_type_classifier")
    strong_sort_dir = Path(home_dir/"projects/StrongSORT")
    fast_re_id_dir = Path(home_dir/"projects/fast-reid")
    logging_dir = Path(output_dir/"dataset_statistics")

class DetectionPaths:    
    person_detections_dir = Path(BasePaths.home_dir/"outputs/yolov8/")
    face_detections_dir = Path(BasePaths.home_dir/"outputs/mtcnn/")
    #videos_input_dir = Path(BasePaths.data_dir//"videos_train/") 
    quantex_videos_input_dir = Path(BasePaths.data_dir/"quantex_videos/") 
    childlens_videos_input_dir = Path(BasePaths.data_dir/"childlens_videos/") 
    images_input_dir = Path(BasePaths.data_dir/"quantex_videos_processed/")
    childlens_images_input_dir = Path(BasePaths.data_dir/"childlens_videos_processed/")
    face_images_input_dir = Path(BasePaths.data_dir/"quantex_rawframes_face/")
    gaze_images_input_dir = Path(BasePaths.data_dir/"quantex_rawframes_gaze/")
    person_images_input_dir = Path(BasePaths.data_dir/"quantex_rawframes_person/")
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

    yolo_detections_dir = Path(BasePaths.leuphana_ipe_dir/"src/models/yolo_detections/")
    
    all_trained_weights_path = Path(BasePaths.models_dir/'yolo11_all_detection.pt')
    
    object_trained_weights_path = Path(BasePaths.models_dir/'yolo11_object_detection.pt')
    object_data_config_path = yolo_detections_dir/"object_dataset.yaml"
    object_labels_input_dir = Path(BasePaths.data_dir/"object_det_labels")
    object_data_input_dir = Path(BasePaths.data_dir/"object_det_input")
    object_output_dir = Path(BasePaths.output_dir/"object_detections/")
    
    person_face_trained_weights_path = Path(BasePaths.models_dir/'yolo11_person_face_detection.pt')
    person_face_data_config_path = yolo_detections_dir/"person_face_dataset.yaml"
    person_face_labels_input_dir = Path(BasePaths.data_dir/"person_face_det_labels")
    person_face_data_input_dir = Path(BasePaths.data_dir/"person_face_det_input")
    person_face_output_dir = Path(BasePaths.output_dir/"person_face_detections/")
    
    all_trained_weights_path = Path(BasePaths.models_dir/'yolo11_all_detection.pt')
    all_data_config_path = yolo_detections_dir/"all_dataset.yaml"
    all_labels_input_dir = Path(BasePaths.data_dir/"all_det_labels")
    all_data_input_dir = Path(BasePaths.data_dir/"all_det_input")
    all_output_dir = Path(BasePaths.output_dir/"all_detections/")
    
    person_face_cls_classes = ['child_person_face', 'adult_person_face']

    @classmethod
    def get_target_paths(cls, target: str, split_type: str) -> Optional[Tuple[Path, Path]]:
        """
        Get image and label destination paths for a given target and split type.
        
        Parameters
        ----------
        target : str 
            The specific class to get paths for (e.g., 'gaze', 'child_person_face', 'object')
        split_type : str
            The dataset split ('train', 'val', or 'test')
                
        Returns
        -------
        Optional[Tuple[Path, Path]]
            Tuple of (images_path, labels_path) for the requested target class
        
        Raises
        ------
        ValueError
            If split_type is not one of 'train', 'val', 'test'
        """
        # Validate split type
        valid_splits = {'train', 'val', 'test'}
        if split_type not in valid_splits:
            raise ValueError(f"Invalid split_type: {split_type}. Must be one of {valid_splits}")

        # Define path mappings for different targets
        path_mappings = {            
            # Object detection paths
            'object': (cls.object_data_input_dir / "images" / split_type,
                    cls.object_data_input_dir / "labels" / split_type),
            'all': (cls.all_data_input_dir / "images" / split_type,
                                cls.all_data_input_dir / "labels" / split_type),
            'person_face': (cls.person_face_data_input_dir / "images" / split_type,
                        cls.person_face_data_input_dir / "labels" / split_type)
        }

        # Return paths if target exists
        if target in path_mappings:
            return path_mappings[target]

        return None

class ClassificationPaths:
    person_classes = ['child_person', 'adult_person']
    person_extracted_faces_dir = Path(BasePaths.data_dir/"person_cls_input")
    person_trained_weights_path = Path(BasePaths.models_dir/'yolo11_person_classification.pt')
    person_extraction_progress_file_path = Path(BasePaths.data_dir/"person_cls_extraction_progress.txt")
    person_missing_frames_file_path = Path(BasePaths.data_dir/"person_cls_missing_frames.txt")
    person_labels_input_dir = Path(BasePaths.data_dir/"person_cls_labels")
    person_data_input_dir = Path(BasePaths.data_dir/"person_cls_input")
    person_output_dir = Path(BasePaths.output_dir/"person_classification/")
    person_data_config_path = Path(BasePaths.leuphana_ipe_dir/"src/models/yolo_classifications/person_dataset.yaml")

    face_classes = ['child_face', 'adult_face']
    face_extracted_faces_dir = Path(BasePaths.data_dir/"face_cls_input")
    face_trained_weights_path = Path(BasePaths.models_dir/'yolo11_face_classification.pt')
    face_extraction_progress_file_path = Path(BasePaths.data_dir/"face_extraction_progress.txt")
    face_missing_frames_file_path = Path(BasePaths.data_dir/"face_missing_frames.txt")
    face_labels_input_dir = Path(BasePaths.data_dir/"face_cls_labels")
    face_data_input_dir = Path(BasePaths.data_dir/"face_cls_input")
    face_output_dir = Path(BasePaths.output_dir/"face_classification/")
    face_data_config_path = Path(BasePaths.leuphana_ipe_dir/"src/models/yolo_classifications/face_dataset.yaml")
    
    gaze_classes = ['no_gaze', 'gaze']
    gaze_extracted_faces_dir = Path(BasePaths.data_dir/"gaze_cls_input")
    gaze_trained_weights_path = Path(BasePaths.models_dir/'yolo11_gaze_classification.pt')
    gaze_extraction_progress_file_path = Path(BasePaths.data_dir/"gaze_extraction_progress.txt")
    gaze_missing_frames_file_path = Path(BasePaths.data_dir/"gaze_missing_frames.txt")
    gaze_data_config_path = Path(BasePaths.leuphana_ipe_dir/"src/models/yolo_gaze_classification/gaze_dataset.yaml")
    gaze_labels_input_dir = Path(BasePaths.data_dir/"gaze_cls_labels")
    gaze_data_input_dir = Path(BasePaths.data_dir/"gaze_cls_input")
    gaze_output_dir = Path(BasePaths.output_dir/"resnet_gaze_classification/")
    
    @classmethod
    def get_target_paths(cls, target: str, split_type: str) -> Optional[Tuple[Path, Path]]:
        """
        Get image and label destination paths for a given target and split type.
        
        Parameters
        ----------
        target : str 
            The specific class to get paths for (e.g., 'child_face', 'adult_person', 'gaze')
        split_type : str
            The dataset split ('train', 'val', or 'test')
            
        Returns
        -------
        Optional[Tuple[Path, Path]]
            Tuple of (input_path, output_path) for the requested target class
        """
        path_mapping = {}
        
        # Add face class paths if target is face-related
        if target in cls.face_classes:
            return (
                cls.face_data_input_dir / split_type / target,
                cls.face_data_input_dir / split_type / target
            )
                
        # Add person class paths if target is person-related
        if target in cls.person_classes:
            return (
                cls.person_data_input_dir / split_type / target,
                cls.person_data_input_dir / split_type / target
            )
        
        if target in cls.gaze_classes:
            return (
                cls.gaze_data_input_dir / split_type / target,
                cls.gaze_data_input_dir / split_type / target
            )

        return None  # Return None if target is not found
    
class VTCPaths:
    childlens_audio_dir = Path(BasePaths.data_dir/"childlens_audio")
    quantex_audio_dir = Path(BasePaths.data_dir/"quantex_audio")

    vtc_output_dir = Path(BasePaths.output_dir/"vtc/")
    vtc_finetuned_output_dir = Path(BasePaths.output_dir/"vtc_finetuned/")
    vtc_from_scratch_output_dir = Path(BasePaths.output_dir/"vtc_from_scratch/")
    
    quantex_output_dir = Path(vtc_output_dir/"quantex_audio")
    childlens_output_dir = Path(vtc_output_dir/"childlens_audio")
    childlens_df_file_path_01 = Path(vtc_output_dir/"childlens_df_duration_off_01.pkl")
    childlens_df_file_path_02 = Path(vtc_output_dir/"childlens_df_duration_off_02.pkl")
    childlens_df_file_path_20 = Path(vtc_output_dir/"childlens_df_duration_off_20.pkl")
    childlens_gt_df_file_path = Path(DetectionPaths.childlens_annotations_dir/"processed/childlens_annotations_gt.pkl")



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
    quantex_processed_videos_log = Path(BasePaths.data_dir/"quantex_processed_videos.txt")
    childlens_processed_videos_log = Path(BasePaths.data_dir/"childlens_processed_videos.txt")
    quantex_rawframes_extraction_error_log = Path(BasePaths.data_dir/"quantex_rawframes_extraction_error.log")
    childlens_rawframes_extraction_error_log = Path(BasePaths.data_dir/"childlens_rawframes_extraction_error.log")
    
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

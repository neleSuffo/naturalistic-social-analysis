import os
import cv2
import logging
import argparse
import numpy as np
from ultralytics import YOLO
from supervision import Detections
from pathlib import Path
from typing import Tuple
from PIL import Image
from constants import ClassificationPaths

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def load_model(model_path: Path = YoloPaths.person_trained_weights_path) -> YOLO:
    """Load YOLO model from path"""
    try:
        model = YOLO(str(model_path))
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

def process_image_classification(model: YOLO, image_path: Path) -> Tuple[int, float, Dict[int, str]]:
    """
    Process an image with a YOLO classification model.

    Returns:
        Tuple containing:
            - top1_idx (int): Index of the top predicted class.
            - top1_conf (float): Confidence of the top predicted class.
            - class_names (Dict[int, str]): Dictionary mapping class indices to names.
    """
    try:
        image_pil = Image.open(image_path)
        results = model(image_pil, verbose=False)

        if not results or not hasattr(results[0], 'probs'):
            raise ValueError(f"Model did not return expected classification results for {image_path.name}.")

        probs = results[0].probs
        top1_idx = probs.top1
        top1_conf = float(probs.top1conf)
        class_names_dict = results[0].names

        logging.info(f"Image: {image_path.name}, Predicted class_idx: {top1_idx} ({class_names_dict.get(top1_idx, 'Unknown')}), Confidence: {top1_conf:.4f}")
        return top1_idx, top1_conf, class_names_dict
    except Exception as e:
        logging.error(f"Error processing image {image_path.name}: {e}")
        raise

def get_ground_truth_from_label_file(
    image_path: Path,
    target: str,
    class_names_dict: Dict[int, str]
) -> Optional[Tuple[int, str]]:
    """
    Infers the ground truth class from the corresponding label file and detection index.
    Example image_path: .../quantex_at_home_id261609_2022_04_01_01_000000_person_2.jpg
    Looks for label in: .../labels_dir/quantex_at_home_id261609_2022_04_01_01_000000.txt, line 2 (0-indexed)
    """
    try:
        image_filename_stem = image_path.stem # e.g., quantex_at_home_id261609_2022_04_01_01_000000_person_2
        parts = image_filename_stem.split('_')

        if len(parts) < 3 or not parts[-1].isdigit() or not parts[-2] in ["person", "face", "gaze"]: # basic check
            logging.warning(f"Filename {image_path.name} does not match expected format for index extraction (e.g., ..._type_index).")
            return None

        detection_index_str = parts[-1]
        detection_type = parts[-2] # "person", "face", etc.
        original_base_name = "_".join(parts[:-2]) # e.g., quantex_at_home_id261609_2022_04_01_01_000000
        
        try:
            detection_index = int(detection_index_str) # This is 0-indexed for file lines
        except ValueError:
            logging.warning(f"Could not parse detection index from {detection_index_str} in {image_path.name}")
            return None

        # Determine annotation folder based on target
        if target == "person":
            annotation_folder = ClassificationPaths.person_labels_input_dir
        elif target == "face":
            annotation_folder = ClassificationPaths.face_labels_input_dir
        elif target == "gaze":
            annotation_folder = ClassificationPaths.gaze_labels_input_dir
        else:
            logging.error(f"Unknown target '{target}' for determining label directory.")
            return None
            
        if not annotation_folder or not annotation_folder.exists():
            logging.error(f"Annotation folder for target '{target}' not found or not configured: {annotation_folder}")
            return None

        label_file_name = f"{original_base_name}.txt"
        label_file_path = annotation_folder / label_file_name

        if not label_file_path.exists():
            logging.warning(f"Label file not found: {label_file_path}")
            return None

        with open(label_file_path, 'r') as f:
            lines = f.readlines()

        if detection_index < len(lines):
            line_content = lines[detection_index].strip()
            if not line_content:
                logging.warning(f"Line {detection_index} in {label_file_path} is empty.")
                return None
            
            gt_class_id_str = line_content.split()[0]
            try:
                gt_class_id = int(gt_class_id_str)
                gt_class_name = class_names_dict.get(gt_class_id)
                if gt_class_name:
                    logging.info(f"Ground truth for {image_path.name} (index {detection_index}): {gt_class_name} (ID: {gt_class_id}) from {label_file_path}")
                    return gt_class_name
                else:
                    logging.warning(f"Ground truth class ID {gt_class_id} from {label_file_path} not in model's class names: {class_names_dict}")
                    return None
            except ValueError:
                logging.warning(f"Could not parse ground truth class ID '{gt_class_id_str}' from {label_file_path}, line {detection_index}.")
                return None
            except IndexError:
                logging.warning(f"Line {detection_index} in {label_file_path} does not contain class information.")
                return None
        else:
            logging.warning(f"Detection index {detection_index} is out of bounds for label file {label_file_path} with {len(lines)} lines.")
            return None

    except Exception as e:
        logging.error(f"Error getting ground truth for {image_path.name} from label file: {e}", exc_info=True)
        return None
    return None

def main():
    parser = argparse.ArgumentParser(description='YOLO Image Classification Inference')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image (e.g., .../quantex_at_home_id261609_2022_04_01_01_000000_person_2.jpg)')
    parser.add_argument('--target', type=str, required=True, choices=["gaze", "person", "face"],
                        help="Target classification type (gaze, person, or face) to locate correct label files.")
    
    args = parser.parse_args()

    if target == "gaze":
        model_p = ClassificationPaths.gaze_trained_weights_path
    elif target == "person":
        model_p = ClassificationPaths.person_trained_weights_path
    elif target == "face":
        model_p = ClassificationPaths.face_trained_weights_path
    else:
        logging.error(f"Unknown target '{args.target}' specified. Must be one of: gaze, person, face.")
        return 1
    # Set paths based on targe
    image_p = Path(args.image_path)

    if not model_p.exists():
        logging.error(f"Model file not found: {model_p}")
        return 1
    if not image_p.exists():
        logging.error(f"Image file not found: {image_p}")
        return 1

    try:
        model = load_model(model_p)
        
        pred_class_idx, pred_conf, class_names_dict = process_image_classification(model, image_p)
        predicted_class_name = class_names_dict.get(pred_class_idx, f"Unknown_Index_{pred_class_idx}")

        gt_class_id, gt_class_name = None, None
        gt_res = get_ground_truth_from_label_file(image_p, args.target, class_names_dict)
        if gt_res and gt_res[1] is not None: # Check if name was found
            gt_class_id, gt_class_name = gt_res
        
        print(f"Predicted label: {predicted_class_name}, Ground truth: {gt_class_name if gt_class_name else 'Not found/Error'}")

        if gt_class_name:
            if pred_class_idx == gt_class_id:
                logging.info(f"Prediction: CORRECT")
            else:
                logging.info(f"Prediction: INCORRECT")
        else:
            logging.info(f"Ground truth could not be determined for {image_p.name}.")

    except Exception as e:
        logging.error(f"Processing failed for {image_p.name}: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
import os
import cv2
import logging
import shutil
from pathlib import Path
from typing import List, Tuple
from ultralytics import YOLO
from supervision import Detections  # Add this import
import numpy as np
from constants import YoloPaths

def load_model(model_path: Path) -> YOLO:
    """Load YOLO model from path"""
    try:
        model = YOLO(str(model_path))
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

def process_image(model: YOLO, image_path: Path) -> Tuple[np.ndarray, Detections]:
    """Process image with YOLO model"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    output = model.predict(image, save=False)
    results = Detections.from_ultralytics(output[0])
    return image, results

def draw_detections(image: np.ndarray, results: Detections) -> np.ndarray:
    """Draw bounding boxes and labels on image"""
    annotated_image = image.copy()
    for bbox, conf in zip(results.xyxy, results.confidence):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Face: {conf:.2f}"
        cv2.putText(annotated_image, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return annotated_image

def load_ground_truth(label_path: str, img_width: int, img_height: int) -> np.ndarray:
    """Load ground truth bounding boxes from label file"""
    ground_truth_boxes = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            # Parse the label file: class x_center y_center width height
            class_id, x_center, y_center, width, height = map(float, line.strip().split())

            # Convert normalized coordinates to pixel values
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)
            
            ground_truth_boxes.append(np.array([x1, y1, x2, y2]))
    
    return np.array(ground_truth_boxes)

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU) between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Compute the area of the intersection rectangle
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Compute the area of both rectangles
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Compute the area of the union
    union = area1 + area2 - intersection
    
    # Compute the IoU
    return intersection / union if union > 0 else 0

def handle_false_positives(predicted_boxes, matched_predictions, misclassified_file, image_path, iou_threshold):
    """Handle false positive cases and write them to the misclassified file"""
    for idx, predicted_box in enumerate(predicted_boxes):
        if idx not in matched_predictions:
            # Check IoU with all ground truth boxes
            max_iou = 0
            for gt_box in predicted_boxes:
                iou = calculate_iou(predicted_box, gt_box)
                max_iou = max(max_iou, iou)
            cv2.imwrite(str(false_positives_dir / image_path.name), image)
            misclassified_file.write(f"{image_path.name} (False Positive) - IoU: {max_iou:.4f}\n")

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    test_dir = Path("/home/nele_pauline_suffo/ProcessedData/yolo_face_input/images/test")
    labels_dir = Path("/home/nele_pauline_suffo/ProcessedData/yolo_face_input/labels/test")
    output_dir = Path("/home/nele_pauline_suffo/outputs/yolo_face_detections")
    output_dir.mkdir(parents=True, exist_ok=True)
    correct_dir = output_dir / "correct"
    true_positives_dir = output_dir / "true_positives"
    true_negatives_dir = output_dir / "true_negatives"
    false_positives_dir = output_dir / "false_positives"
    false_negatives_dir = output_dir / "false_negatives"
    # delete correct_dir and misclassified_dir if they already exist
    shutil.rmtree(true_positives_dir, ignore_errors=True)
    shutil.rmtree(true_negatives_dir, ignore_errors=True)
    shutil.rmtree(false_positives_dir, ignore_errors=True)
    shutil.rmtree(false_negatives_dir, ignore_errors=True)
    true_positives_dir.mkdir(parents=True, exist_ok=True)
    true_negatives_dir.mkdir(parents=True, exist_ok=True)
    false_positives_dir.mkdir(parents=True, exist_ok=True)
    false_negatives_dir.mkdir(parents=True, exist_ok=True)

    correct_txt = output_dir / "correct.txt"
    misclassified_txt = output_dir / "misclassified.txt"
    # delete correct.txt and misclassified.txt if they already exist
    correct_txt.unlink(missing_ok=True)
    misclassified_txt.unlink(missing_ok=True)

    model = load_model(YoloPaths.face_trained_weights_path)

    iou_threshold = 0.5

    # Initialize counters
    true_positive_count = 0
    true_negative_count = 0
    false_positive_count = 0
    false_negative_count = 0

    # Open text files for writing
    with open(correct_txt, 'w') as correct_file, open(misclassified_txt, 'w') as misclassified_file:
        for image_path in test_dir.glob("*.jpg"):
            try:
                label_path = labels_dir / (image_path.stem + ".txt")
                image, results = process_image(model, image_path)
                predicted_boxes = results.xyxy
                
                # Get image dimensions for ground truth conversion
                img_height, img_width = image.shape[:2]
                ground_truth_boxes = load_ground_truth(str(label_path), img_width, img_height)

                num_faces_in_image = len(ground_truth_boxes)  # Number of ground truth faces
                num_predicted_faces = len(predicted_boxes)  # Number of predicted faces

                # No ground truth faces at all
                if num_faces_in_image == 0:  # Check if there are no ground truth faces
                    if num_predicted_faces > 0:  # False positive
                        false_positive_count += 1
                        cv2.imwrite(str(false_positives_dir / image_path.name), image)
                        misclassified_file.write(f"{image_path.name} (False Positive) - IoU: N/A\n")
                    else:  # True negative (no faces in both)
                        true_negative_count += 1
                        cv2.imwrite(str(true_negatives_dir / image_path.name), image)
                        correct_file.write(f"{image_path.name} (True Negative)\n")
                    continue

                # For each face in the image, treat them as separate classification cases
                matched_predictions = set()
                matched_ground_truths = set()

                for pred_idx, detected_bbox in enumerate(predicted_boxes):
                    for gt_idx, gt_bbox in enumerate(ground_truth_boxes):
                        iou = calculate_iou(detected_bbox, gt_bbox)
                        logging.info(f"Detection IoU: {iou:.4f}")
                        if iou >= iou_threshold:
                            matched_predictions.add(pred_idx)
                            matched_ground_truths.add(gt_idx)

                            # Write the IoU value for matched detections
                            correct_file.write(f"{image_path.name} (True Positive) - IoU: {iou:.4f}\n")

                # Handle ground truth faces that were not detected (False Negatives)
                unmatched_ground_truths = set(range(num_faces_in_image)) - matched_ground_truths
                false_negative_count += len(unmatched_ground_truths)
                for _ in unmatched_ground_truths:
                    cv2.imwrite(str(false_negatives_dir / image_path.name), image)
                    misclassified_file.write(f"{image_path.name} (False Negative)\n")

                # Handle faces that were predicted but not matched to ground truths (False Positives)
                unmatched_predictions = set(range(num_predicted_faces)) - matched_predictions
                false_positive_count += len(unmatched_predictions)
                handle_false_positives(predicted_boxes, matched_predictions, misclassified_file, image_path, iou_threshold)

                # Handle True Positives
                true_positive_count += len(matched_ground_truths)
                for idx in matched_predictions:
                    annotated_image = draw_detections(image, results)
                    cv2.imwrite(str(true_positives_dir / image_path.name), annotated_image)

                logging.info(f"Processed {image_path.name}")
            except Exception as e:
                logging.error(f"Error processing {image_path}: {e}")

    # Log final results
    logging.info(f"True Positives: {true_positive_count}")
    logging.info(f"True Negatives: {true_negative_count}")
    logging.info(f"False Positives: {false_positive_count}")
    logging.info(f"False Negatives: {false_negative_count}")

    # Save final counts to text files
    with open(output_dir / "summary.txt", "w") as summary_file:
        summary_file.write(f"True Positives: {true_positive_count}\n")
        summary_file.write(f"True Negatives: {true_negative_count}\n")
        summary_file.write(f"False Positives: {false_positive_count}\n")
        summary_file.write(f"False Negatives: {false_negative_count}\n")

    logging.info("Evaluation complete. Results saved.")
    
if __name__ == "__main__":
    main()

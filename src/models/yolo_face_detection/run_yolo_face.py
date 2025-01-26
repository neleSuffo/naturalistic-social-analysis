import os
import cv2
import logging
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


def parse_annotations(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Parse annotations from YOLO label file"""
    annotations = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:  # Class, x_center, y_center, width, height
                annotations.append(tuple(map(float, parts)))
    return annotations


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU) between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


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
    misclassified_dir = output_dir / "misclassified"
    correct_dir.mkdir(parents=True, exist_ok=True)
    misclassified_dir.mkdir(parents=True, exist_ok=True)

    correct_txt = output_dir / "correct.txt"
    misclassified_txt = output_dir / "misclassified.txt"

    model = load_model(YoloPaths.face_trained_weights_path)

    iou_threshold = 0.5

    # Initialize counters
    true_positive_count = 0
    true_negative_count = 0
    false_positive_count = 0
    false_negative_count = 0

    # Open text files for writing
    with open(correct_txt, 'w') as correct_file, open(misclassified_txt, 'w') as misclassified_file:
        logging.info(f"Found {len(test_dir.glob('*.jpg'))} images in test directory")
        for image_path in test_dir.glob("*.jpg"):
            try:
                label_path = labels_dir / (image_path.stem + ".txt")
                print("Label path: ", label_path)
                ground_truths = parse_annotations(label_path)  # Parse annotations
                print("Ground truths: ", ground_truths)

                image, results = process_image(model, image_path)
                predicted_boxes = results.xyxy
                print("Predicted boxes: ", predicted_boxes)

                if not ground_truths:  # No ground truth annotations
                    if len(predicted_boxes) > 0:  # False positive
                        false_positive_count += 1
                        cv2.imwrite(str(misclassified_dir / image_path.name), image)
                        misclassified_file.write(f"{image_path.name} (False Positive)\n")
                    else:  # True negative
                        true_negative_count += 1
                        cv2.imwrite(str(correct_dir / image_path.name), image)
                        correct_file.write(f"{image_path.name} (True Negative)\n")
                    continue

                # Evaluate predictions against ground truths
                matched_predictions = set()
                matched_ground_truths = set()

                for pred_idx, pred_box in enumerate(predicted_boxes):
                    pred_box = pred_box[:4].astype(float)  # Extract bounding box
                    pred_x1, pred_y1, pred_x2, pred_y2 = pred_box

                    for gt_idx, gt in enumerate(ground_truths):
                        _, x_center, y_center, width, height = gt
                        gt_x1 = x_center - width / 2
                        gt_y1 = y_center - height / 2
                        gt_x2 = x_center + width / 2
                        gt_y2 = y_center + height / 2

                        iou = calculate_iou(pred_box, [gt_x1, gt_y1, gt_x2, gt_y2])
                        if iou >= iou_threshold:
                            matched_predictions.add(pred_idx)
                            matched_ground_truths.add(gt_idx)

                # True Positives
                if matched_predictions and matched_ground_truths:
                    true_positive_count += len(matched_ground_truths)
                    annotated_image = draw_detections(image, results)
                    cv2.imwrite(str(correct_dir / image_path.name), annotated_image)
                    correct_file.write(f"{image_path.name} (True Positive)\n")

                # False Positives: Predictions not matched to any ground truth
                unmatched_predictions = set(range(len(predicted_boxes))) - matched_predictions
                if unmatched_predictions:
                    false_positive_count += len(unmatched_predictions)
                    cv2.imwrite(str(misclassified_dir / image_path.name), image)
                    misclassified_file.write(f"{image_path.name} (False Positive)\n")

                # False Negatives: Ground truths not matched to any predictions
                unmatched_ground_truths = set(range(len(ground_truths))) - matched_ground_truths
                if unmatched_ground_truths:
                    false_negative_count += len(unmatched_ground_truths)
                    cv2.imwrite(str(misclassified_dir / image_path.name), image)
                    misclassified_file.write(f"{image_path.name} (False Negative)\n")

                logging.info(f"Processed {image_path.name}")
            except Exception as e:
                logging.error(f"Error processing {image_path}: {e}")
            break

    # Log final results
    logging.info(f"True Positives: {true_positive_count}")
    logging.info(f"True Negatives: {true_negative_count}")
    logging.info(f"False Positives: {false_positive_count}")
    logging.info(f"False Negatives: {false_negative_count}")

    logging.info("Evaluation complete. Results saved.")
    
    
if __name__ == "__main__":
    main()
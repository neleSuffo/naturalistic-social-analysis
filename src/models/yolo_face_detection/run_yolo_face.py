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
from huggingface_hub import hf_hub_download
from constants import YoloPaths

def load_model(model_path: Path = YoloPaths.face_trained_weights_path) -> YOLO:
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
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
            
        output = model(Image.open(image_path))
        results = Detections.from_ultralytics(output[0])
        logging.info(f"Detected {len(results.xyxy)} faces")
        return image, results
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise

def draw_detections(image: np.ndarray, results: Detections) -> np.ndarray:
    """Draw bounding boxes and labels on image"""
    annotated_image = image.copy()
    for bbox, conf in zip(results.xyxy, results.confidence):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Face: {conf:.2f}"
        cv2.putText(annotated_image, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return annotated_image

def calculate_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of the intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the area of the union
    unionArea = boxAArea + boxBArea - interArea

    # Compute the IoU
    return interArea / unionArea if unionArea > 0 else 0

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

def main():
    parser = argparse.ArgumentParser(description='YOLO Face Detection')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    args = parser.parse_args()  
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Initialize paths
    output_dir = YoloPaths.face_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load model and process image
        model = load_model(YoloPaths.face_trained_weights_path)
        image, results = process_image(model, Path(args.image))
        
        # Get image dimensions for ground truth conversion
        img_height, img_width = image.shape[:2]
        
        # Load ground truth labels
        label_path = args.image.replace("images", "labels").replace(".jpg", ".txt")
        ground_truth_boxes = load_ground_truth(label_path, img_width, img_height)
        
        # Loop through detections and calculate IoU
        for i, detected_bbox in enumerate(results.xyxy):
            iou_scores = []
            for gt_bbox in ground_truth_boxes:
                iou = calculate_iou(detected_bbox, gt_bbox)
                iou_scores.append(iou)
            max_iou = max(iou_scores)
            logging.info(f"Detected face {i+1} - Maximum IoU: {max_iou:.4f}")
        
        # Draw detections and save
        annotated_image = draw_detections(image, results)
        output_path = output_dir / Path(args.image).name
        cv2.imwrite(str(output_path), annotated_image)
        logging.info(f"Annotated image saved to: {output_path}")
        
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
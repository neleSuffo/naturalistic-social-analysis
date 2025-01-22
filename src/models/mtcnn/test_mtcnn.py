import os
import json
import numpy as np
import torch
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from facenet_pytorch import MTCNN
from PIL import Image
from constants import MtcnnPaths, DetectionPaths
from sklearn.metrics import precision_score, recall_score, f1_score

# Function to parse labels.txt
def parse_labels_file(labels_file_path):
    annotations = {}
    with open(labels_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            image_path = parts[0]
            bbox = list(map(float, parts[1].split(',')))  # x1, y1, width, height, label
            annotations[image_path] = bbox
    return annotations

def classify_detections(gt_bbox, detected_boxes, iou_threshold=0.5):
    """
    Classify detections as TP, FP, or FN based on IoU threshold.

    Args:
        gt_bbox: Ground truth bounding box [x, y, w, h].
        detected_boxes: List of detected bounding boxes.
        iou_threshold: IoU threshold to consider a match.

    Returns:
        tp: True positives.
        fp: False positives.
        fn: False negatives.
    """
    matched = False
    tp, fp = 0, 0

    for detected_box in detected_boxes:
        detected_box = [
            detected_box[0],
            detected_box[1],
            detected_box[2] - detected_box[0],  # width
            detected_box[3] - detected_box[1],  # height
        ]
        iou = calculate_iou(gt_bbox, detected_box)
        if iou >= iou_threshold:
            matched = True
            tp += 1
        else:
            fp += 1

    fn = 1 if not matched else 0
    return tp, fp, fn

def extract_video_name(image_name):
    """
    Extracts video name from image filename by removing frame number and extension
    
    Args:
        image_name (str): Full image filename (e.g. 'quantex_at_home_id257511_2021_07_12_02_001200.jpg')
    
    Returns:
        str: Video name without frame number
    """
    # Remove last 7 characters (_XXXXXX.jpg)
    video_name = image_name[:-11]
    return video_name

# Function to calculate Intersection over Union (IoU)
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def plot_confusion_matrix(tp, fp, fn, tn, output_path):
    """Plot and save confusion matrix"""
    cm = np.array([[tn, fp],
                   [fn, tp]])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Face', 'Face'],
                yticklabels=['No Face', 'Face'])
    plt.title('MTCNN Face Detection Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_path)
    plt.close()
   
def get_matrix_path(output_path):
    """Create path for confusion matrix plot"""
    base_path = os.path.splitext(str(output_path))[0]
    return f"{base_path}_confusion_matrix.png"
 
# Updated Evaluation Function
def evaluate_mtcnn_with_metrics(labels_file_path, images_folder, output_path):
    annotations = parse_labels_file(labels_file_path)
    mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

    results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config": {
            "device": 'cuda' if torch.cuda.is_available() else 'cpu',
            "labels_file": str(labels_file_path),  # Convert to string
            "images_folder": str(images_folder)     # Convert to string
        },
        "detections": {},
        "metrics": {}
    }

    iou_scores = []
    tp_total, fp_total, fn_total = 0, 0, 0

    # Process each image
    for image_name, gt_bbox in annotations.items():
        video_subfolder = extract_video_name(image_name)
        image_path = os.path.join(images_folder, video_subfolder, image_name)
    
        # Validate image path
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Load and process image
        img = Image.open(image_path).convert('RGB')
        boxes, _ = mtcnn.detect(img)
        
        # Skip if no face detected
        if boxes is None:
            fn_total += 1
            results["detections"][image_name] = {
                "detected": False,
                "ground_truth": gt_bbox,
                "detected_boxes": []
            }
            continue
        
        # Calculate metrics for this image
        tp, fp, fn = classify_detections(gt_bbox, boxes, iou_threshold=0.5)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        
        # Calculate IoU for detected boxes
        max_iou = 0
        for box in boxes:
            detected_box = [
                box[0],
                box[1],
                box[2] - box[0],
                box[3] - box[1]
            ]
            iou = calculate_iou(gt_bbox, detected_box)
            max_iou = max(max_iou, iou)
        
        if max_iou > 0:
            iou_scores.append(max_iou)
        
        # Store detection results
        results["detections"][image_name] = {
            "detected": boxes is not None,
            "ground_truth": gt_bbox,
            "detected_boxes": boxes.tolist() if boxes is not None else [],
            "iou": max_iou
        }

    # Calculate final metrics
    total_predictions = tp_total + fp_total
    total_ground_truth = tp_total + fn_total
    
    # Calculate true negatives (images without faces that were correctly identified)
    tn_total = len([boxes for boxes in results["detections"].values() 
                    if not boxes["detected"] and not any(boxes["ground_truth"])])
    
    precision = tp_total / total_predictions if total_predictions > 0 else 0
    recall = tp_total / total_ground_truth if total_ground_truth > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    average_iou = np.mean(iou_scores) if len(iou_scores) > 0 else 0

    # Store metrics
    results["metrics"] = {
        "average_iou": float(f"{average_iou:.4f}"),
        "precision": float(f"{precision:.4f}"),
        "recall": float(f"{recall:.4f}"),
        "f1_score": float(f"{f1:.4f}"),
        "true_positives": tp_total,
        "true_negatives": tn_total,
        "false_positives": fp_total,
        "false_negatives": fn_total,
    }

    # Plot confusion matrix
    matrix_path = get_matrix_path(output_path)
    plot_confusion_matrix(tp_total, fp_total, fn_total, tn_total, matrix_path)
    
    # Save results to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

# Paths
labels_file_path = MtcnnPaths.labels_file_path
images_folder = DetectionPaths.images_input_dir
output_path = MtcnnPaths.face_detection_results_file_path

# Run evaluation
results = evaluate_mtcnn_with_metrics(
    labels_file_path, 
    images_folder,
    output_path
)
from ultralytics import YOLO
from sklearn.metrics import precision_recall_curve, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
from constants import YoloPaths
import numpy as np
import os

def save_plot(plt, filename):
    """Saves the plot to the specified filename."""
    plt.savefig(filename)
    print(f"Plot saved to: {filename}")
    
def plot_precision_recall_curve(true_labels, pred_scores, output_dir):
    """Plots the Precision-Recall Curve and saves it."""
    precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid()
    
    # Save plot
    save_plot(plt, os.path.join(output_dir, "precision_recall_curve.png"))

def plot_roc_curve(true_labels, pred_scores, output_dir):
    """Plots the ROC Curve and computes AUC, then saves it."""

    fpr, tpr, _ = roc_curve(true_labels, pred_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()

    # Save plot
    save_plot(plt, os.path.join(output_dir, "roc_curve.png"))

def calculate_iou(pred_box, gt_box):
    """
    Calculate Intersection over Union (IoU) between predicted and ground truth boxes.
    
    Args:
        pred_box (list): Predicted bounding box [x_min, y_min, x_max, y_max].
        gt_box (list): Ground truth bounding box [x_min, y_min, x_max, y_max].

    Returns:
        float: IoU score between 0 and 1.
    """
    x_min_pred, y_min_pred, x_max_pred, y_max_pred = pred_box
    x_min_gt, y_min_gt, x_max_gt, y_max_gt = gt_box

    # Calculate intersection area
    x_min_inter = max(x_min_pred, x_min_gt)
    y_min_inter = max(y_min_pred, y_min_gt)
    x_max_inter = min(x_max_pred, x_max_gt)
    y_max_inter = min(y_max_pred, y_max_gt)

    intersection_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)

    # Calculate union area
    pred_area = (x_max_pred - x_min_pred) * (y_max_pred - y_min_pred)
    gt_area = (x_max_gt - x_min_gt) * (y_max_gt - y_min_gt)

    union_area = pred_area + gt_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def load_ground_truths(label_dir, img_names, img_size):
    """
    Load ground truth bounding boxes from YOLO-format labels.
    
    Args:
        label_dir (str): Directory containing the label files.
        img_names (list): List of image filenames.
        img_size (tuple): Size of the images (width, height).
    
    Returns:
        list: List of ground truth bounding boxes for each image.
    """
    ground_truths = []

    for img_name in img_names:
        label_file = os.path.join(label_dir, f"{os.path.splitext(img_name)[0]}.txt")
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                gt_boxes = []
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    # Convert YOLO normalized coordinates to pixel coordinates
                    x_min = (x_center - width / 2) * img_size[0]
                    y_min = (y_center - height / 2) * img_size[1]
                    x_max = (x_center + width / 2) * img_size[0]
                    y_max = (y_center + height / 2) * img_size[1]
                    gt_boxes.append([x_min, y_min, x_max, y_max])
            ground_truths.append(gt_boxes)
        else:
            # If no labels for an image, append an empty list (no faces in this image)
            ground_truths.append([])

    return ground_truths

def calculate_recall_at_threshold(predictions, ground_truths, threshold=0.5):
    """
    Calculate recall at a fixed confidence threshold (e.g., 0.5) for face detection.
    
    Args:
        predictions (list): List of prediction dictionaries with 'boxes' and 'scores' (confidence).
        ground_truths (list): List of ground truth bounding boxes.
        threshold (float): Confidence threshold for determining if a prediction is valid.

    Returns:
        float: Recall at the specified threshold.
    """
    # Initialize variables for True Positives and False Negatives
    true_positives = 0
    false_negatives = 0

    # Iterate over the dataset
    for pred in predictions:
        pred_bboxes = pred.boxes.xywh  # Get bounding boxes
        pred_scores = pred.probs  # Get confidence scores
        
        # Filter predictions based on the threshold
        valid_preds = [box for box, score in zip(pred_bboxes, pred_scores) if score >= threshold]

        for pred in valid_preds:
            iou_max = 0
            for gt in ground_truths:
                iou = calculate_iou(pred, gt)  # IoU function to calculate overlap
                iou_max = max(iou_max, iou)
            if iou_max >= 0.5:
                true_positives += 1
        
        # False negatives: if a ground truth face was not detected
        false_negatives += len(ground_truths) - true_positives
    
    # Calculate recall
    recall = true_positives / (true_positives + false_negatives)
    return recall
       
# Path to the best model weights and test dataset
best_model_path = str(YoloPaths.face_output_dir) + "/yolo_face_finetune7/weights/best.pt"
test_data_path = "/home/nele_pauline_suffo/ProcessedData/yolo_face_input/images/test"
label_dir = "/home/nele_pauline_suffo/ProcessedData/yolo_face_input/labels/val"
output_dir = str(YoloPaths.face_output_dir) + "/yolo_face_finetune7"  # Use the same output directory

model = YOLO(best_model_path)

# Get predictions on the test dataset
predictions = model.predict(test_data_path)

# Load ground truth bounding boxes from labels/val
img_names = os.listdir(test_data_path)  # Assuming test_data_path contains the images
img_size = (640, 640)  # Set the image size used during training
ground_truths = load_ground_truths(label_dir, img_names, img_size)

# Calculate recall at threshold 0.5
recall = calculate_recall_at_threshold(predictions, ground_truths, threshold=0.5)
print(f"Recall at 0.5 confidence threshold: {recall:.4f}")
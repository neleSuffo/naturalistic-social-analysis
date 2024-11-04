import torch
import cv2
from pathlib import Path
from ultralytics import YOLO
from strong_sort import StrongSORT
from constants import YoloPaths

# Initialize YOLO model
yolo_model = YOLO(YoloPaths.trained_weights_path) 

# Initialize StrongSORT tracker
tracker = StrongSORT(model_weights='model.pth', device='cuda')

# Directory containing images
image_dir = Path('/home/nele_pauline_suffo/ProcessedData/strong_sort/quantex/test/quantex_at_home_id258239_2020_08_23_01/img1')

# Iterate through each image in the directory
for image_path in sorted(image_dir.glob('*.jpg')):  # Adjust the file extension if needed
    # Load image
    img = cv2.imread(str(image_path))

    # Run inference with YOLO model
    with torch.no_grad():
        pred = yolo_model(img)  # Perform detection using YOLO

    # Iterate through detections and update tracker
    for detection in pred:
        # Assuming detection format is (x1, y1, x2, y2, confidence, class_id)
        # Convert YOLO detections to the format expected by StrongSORT
        detections = [det[:4] for det in detection]  # Extract bounding boxes
        for det in detections:
            det = [int(coord) for coord in det]  # Convert to int if needed
            updated_track = tracker.update(det, img)

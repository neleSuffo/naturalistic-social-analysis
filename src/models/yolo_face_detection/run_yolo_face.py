import os
import cv2
import logging
import numpy as np
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
from huggingface_hub import hf_hub_download
from constants import YoloPaths

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create output directory
output_dir = "/home/nele_pauline_suffo/outputs/face_detections"
os.makedirs(output_dir, exist_ok=True)

# Define model storage path
model_file = YoloPaths.face_trained_weights_path
model_dir = model_file.parent
os.makedirs(model_dir, exist_ok=True)

# Download model if not exists
if not model_file.exists():
    logging.info("Downloading YOLOv8 face detection model...")
    model_path = hf_hub_download(
        repo_id="arnabdhar/YOLOv8-Face-Detection", 
        filename="model.pt",
        local_dir=model_dir,
        local_dir_use_symlinks=False
    )
    # Rename downloaded file
    os.rename(model_path, model_file)
    logging.info(f"Model downloaded and saved to: {model_file}")
else:
    logging.info(f"Using existing model from: {model_file}")

model = YOLO(str(model_file))
logging.info("Model loaded successfully")
# Load and process image
image_path = "/home/nele_pauline_suffo/ProcessedData/quantex_annotated_rawframes/quantex_at_home_id262691_2022_02_28_01/quantex_at_home_id262691_2022_02_28_01_043560.jpg"
image = cv2.imread(image_path)
output = model(Image.open(image_path))
results = Detections.from_ultralytics(output[0])
logging.info(f"Detected {len(results.xyxy)} faces")

# Draw bounding boxes
for bbox, conf in zip(results.xyxy, results.confidence):
    x1, y1, x2, y2 = map(int, bbox)
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Add confidence score
    label = f"Face: {conf:.2f}"
    cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save output image
output_path = os.path.join(output_dir, os.path.basename(image_path))
cv2.imwrite(output_path, image)

logging.info(f"Annotated image saved to: {output_path}")
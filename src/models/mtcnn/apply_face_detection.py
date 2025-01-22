import os
import cv2
import numpy as np
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
from huggingface_hub import hf_hub_download

# Create output directory
output_dir = "/home/nele_pauline_suffo/outputs/face_detections"
os.makedirs(output_dir, exist_ok=True)

# Download and load model
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
model = YOLO(model_path)

# Load and process image
image_path = "/home/nele_pauline_suffo/ProcessedData/quantex_annotated_rawframes/quantex_at_home_id255944_2022_03_26_01/quantex_at_home_id255944_2022_03_26_01_003690.jpg"
image = cv2.imread(image_path)
output = model(Image.open(image_path))
results = Detections.from_ultralytics(output[0])

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

print(f"Annotated image saved to: {output_path}")
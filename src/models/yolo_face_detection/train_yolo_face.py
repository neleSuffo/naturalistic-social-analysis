import os
import sys
import torch
import shutil
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from constants import YoloPaths

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Set thread limits
os.environ['OMP_NUM_THREADS'] = '4'  # OpenMP threads
torch.set_num_threads(4)  # PyTorch threads

# Load the YOLO model
model = YOLO("/home/nele_pauline_suffo/models/yolov11_face_detection_AdamCodd.pt")  # Use pretrained YOLOv8 model

# Define experiment name
experiment_name = timestamp + "_yolo_face_finetune_with_augment_and_earlystop"
output_dir = YoloPaths.face_output_dir / experiment_name

# Train the model with a cosine annealing learning rate scheduler
model.train(
    data=str(YoloPaths.face_data_config_path),
    epochs=200,  # Total number of epochs
    imgsz=1280,  # Image size
    batch=16,   # Batch size
    project=str(YoloPaths.face_output_dir),  # Output directory
    name=experiment_name,  # Experiment name
    augment=True,  # Enable YOLO's built-in augmentations
    lr0=0.01,  # Initial learning rate
    lrf=0.001,  # Final learning rate after scheduling
    cos_lr=True,  # Use cosine annealing for learning rate scheduling,
    patience=10,  # Stop training if no improvement for 5 consecutive epochs
    device=0,  # GPU (use "cpu" for CPU training)
    plots=True,  # Plot training results
)

# Save script copy and setup logging after YOLO creates the directory
script_copy = output_dir / "train_yolo_face.py"
shutil.copy(__file__, script_copy)
log_file = output_dir / "output.log"
sys.stdout = open(log_file, 'w', buffering=1)
sys.stderr = sys.stdout
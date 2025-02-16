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
model = YOLO("yolo11m-cls.pt")

# Define experiment name and output directory
experiment_name = timestamp + "_yolo_gaze_finetune_with_augment_and_earlystop"
output_dir = YoloPaths.gaze_output_dir / experiment_name

# Train the model with a cosine annealing learning rate scheduler
model.train(
    data="/home/nele_pauline_suffo/ProcessedData/yolo_gaze_input",
    epochs=200,  # Total number of epochs
    imgsz=1280,  # Image size
    batch=16,   # Batch size
    project=str(YoloPaths.gaze_output_dir),  # Output directory
    name=experiment_name,  # Experiment name
    lr0=0.01,  # Initial learning rate
    lrf=0.001,  # Final learning rate after scheduling
    cos_lr=True,  # Use cosine annealing for learning rate scheduling,
    patience=10,  # Stop training if no improvement for 5 consecutive epochs
    device=0,  # GPU (use "cpu" for CPU training)
    plots=True,  # Plot training results
)

# Copy the script to the output directory after training starts
script_copy = output_dir / "train_yolo_gaze.py"
if os.path.exists(__file__):  # Avoid errors in interactive environments
    shutil.copy(__file__, script_copy)
import os
import torch
from ultralytics import YOLO
from constants import YoloPaths

# Set thread limits
os.environ['OMP_NUM_THREADS'] = '4'  # OpenMP threads
torch.set_num_threads(4)  # PyTorch threads

# Load the YOLO model
model = YOLO("/home/nele_pauline_suffo/models/yolov8_github_face.pt")  # Use pretrained YOLOv8 model

# Train the model with a cosine annealing learning rate scheduler
model.train(
    data=str(YoloPaths.face_data_config_path),
    epochs=100,  # Total number of epochs
    imgsz=640,  # Image size
    batch=16,   # Batch size
    project=str(YoloPaths.face_output_dir),  # Output directory
    name="yolo_face_finetune_with_augment_and_earlystop",  # Experiment name
    augment=True,  # Enable YOLO's built-in augmentations
    lr0=0.01,  # Initial learning rate
    lrf=0.001,  # Final learning rate after scheduling
    cos_lr=True,  # Use cosine annealing for learning rate scheduling,
    patience=5,  # Stop training if no improvement for 5 consecutive epochs
    device=0,  # GPU (use "cpu" for CPU training)
    plots=True,  # Plot training results
)
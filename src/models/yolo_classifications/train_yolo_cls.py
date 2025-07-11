import os
import sys
import torch
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from constants import ClassificationPaths

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train YOLO model for classification")
    parser.add_argument("--target", type=str, required=True, choices=["gaze", "person", "face"],
                      help="Target classification task (gaze, person, or face)")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    target = args.target
    
    # Create timestamp for unique run identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set thread limits
    os.environ['OMP_NUM_THREADS'] = '6'  # OpenMP threads
    torch.set_num_threads(6)  # PyTorch threads

    # Load the YOLO model
    model = YOLO("yolo11x-cls.pt")
    # Define experiment name and output directory
    experiment_name = f"yolo_{target}_cls_{timestamp}"
    output_dir = getattr(ClassificationPaths, f"{target}_output_dir") / experiment_name

    data_dir = f"/home/nele_pauline_suffo/ProcessedData/{target}_cls_input"
    # Train the model with a cosine annealing learning rate scheduler
    model.train(
        data=data_dir,
        epochs=200,  # Total number of epochs
        imgsz=640,  # Image size
        batch=32,   # Batch size
        project=str(getattr(ClassificationPaths, f"{target}_output_dir")),  # Output directory
        name=experiment_name,  # Experiment name
        lr0=0.005,  # Reduce initial learning rate
        lrf=0.0001,  # Final learning rate after scheduling
        weight_decay=0.0005,  # Weight decay
        cos_lr=True,  # Use cosine annealing for learning rate scheduling
        patience=10,  # Stop training if no improvement for 10 consecutive epochs
        amp=True,  # Use automatic mixed precision
        device=0,  # GPU (use "cpu" for CPU training)
        plots=True,  # Plot training results
    )

    # Copy the script to the output directory after training starts
    script_copy = output_dir / f"train_yolo_{target}.py"
    if os.path.exists(__file__):  # Avoid errors in interactive environments
        shutil.copy(__file__, script_copy)

if __name__ == "__main__":
    main()
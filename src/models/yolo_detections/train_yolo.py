import os
import sys
import torch
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from constants import YoloPaths

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO model for different detection tasks')
    parser.add_argument('--yolo_target', type=str, required=True,
                      choices=['person_face', 'adult_person_face', 'child_person_face', 
                              'person_face_object', 'all'],
                      help='Target detection task')
    parser.add_argument('--epochs', type=int, default=200,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=640,
                      help='Image size for training')
    parser.add_argument('--device', type=str, default='0',
                      help='Device to use (e.g., "0" for GPU, "cpu" for CPU)')
    return parser.parse_args()

def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set thread limits
    os.environ['OMP_NUM_THREADS'] = '10'  # OpenMP threads
    torch.set_num_threads(10)  # PyTorch threads

    # Get appropriate paths based on yolo_target
    data_config_path = getattr(YoloPaths, f"{args.yolo_target}_data_config_path")
    base_output_dir = getattr(YoloPaths, f"{args.yolo_target}_output_dir")
    
    # Load the YOLO model
    model = YOLO("yolo11x.pt")

    # Define experiment name and output directory
    experiment_name = f"{timestamp}_yolo_{args.yolo_target}_finetune_with_augment_and_earlystop"
    output_dir = base_output_dir / experiment_name

    # Train the model with a cosine annealing learning rate scheduler
    model.train(
        data=str(data_config_path),
        epochs=args.epochs, # Total number of epochs
        imgsz=args.img_size, # Image size
        batch=args.batch_size, # Batch size
        project=str(base_output_dir), # Output directory
        name=experiment_name, # Experiment name
        augment=True, # Enable YOLO's built-in augmentations
        lr0=0.01, # Initial learning rate
        lrf=0.001, # Final learning rate after scheduling
        cos_lr=True, # Use cosine annealing for learning rate scheduling
        patience=10, # Stop training if no improvement for 5 consecutive epochs
        device=args.device, # GPU (use "cpu" for CPU training)
        plots=True, # Plot training results
    )

    # Copy the script to the output directory after training starts
    script_copy = output_dir / f"train_yolo_{args.yolo_target}.py"
    if os.path.exists(__file__):
        shutil.copy(__file__, script_copy)

if __name__ == "__main__":
    main()
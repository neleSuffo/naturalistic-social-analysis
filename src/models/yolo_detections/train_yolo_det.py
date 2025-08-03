import os
import sys
import torch
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from constants import DetectionPaths

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO model for different detection tasks')
    parser.add_argument('--epochs', type=int, default=300,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=640,
                      help='Image size for training')
    parser.add_argument('--target', type=str, default='all',
                      choices=['all', 'face'],
                      help='Target detection task to train on')
    parser.add_argument('--device', type=str, default='0,1',
                      help='Device to use (e.g., "0" for GPU, "cpu" for CPU)')
    parser.add_argument('--model_size', type=str, default='x',
                      choices=['n', 's', 'm', 'l', 'x'],
                      help='YOLO model size (n=nano, s=small, m=medium, l=large, x=extra-large)')
    parser.add_argument('--pretrained', type=bool, default=True,
                      help='Use pretrained weights')
    parser.add_argument('--resume', type=str, default=None,
                      help='Resume training from checkpoint path')
    return parser.parse_args()

def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set thread limits
    os.environ['OMP_NUM_THREADS'] = '6'  # OpenMP threads
    torch.set_num_threads(6)  # PyTorch threads

    data_config_path = getattr(DetectionPaths, f"{args.target}_data_config_path")
    base_output_dir = getattr(DetectionPaths, f"{args.target}_output_dir")
    
    # Load the YOLO model - try smaller model first if overfitting
    model_name = f"yolo12{args.model_size}.pt"
    print(f"Loading model: {model_name}")
    
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        model = YOLO(args.resume)
    else:
        model = YOLO(model_name)
        
    # Print model info
    print(f"Model parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    print(f"Model size: {args.model_size}")
    print(f"Pretrained: {args.pretrained}")
    
    # For overfitting issues, consider using a smaller model
    if args.model_size == 'x':
        print("WARNING: Using YOLO11x - if overfitting, consider using 'l' or 'm' model size")
    
    # Define experiment name and output directory
    experiment_name = f"{timestamp}_yolo11{args.model_size}_{args.target}"
    output_dir = base_output_dir / experiment_name
    
    print(f"Training will be saved to: {output_dir}")
    print(f"Data config: {data_config_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print("-" * 50)

    # Train the model with improved regularization to reduce overfitting
    model.train(
        data=str(data_config_path),
        epochs=args.epochs, # Total number of epochs
        imgsz=args.img_size, # Image size
        batch=args.batch_size, # Batch size
        project=str(base_output_dir), # Output directory
        name=experiment_name, # Experiment name
        augment=True, # Enable YOLO's built-in augmentations
        
        # Learning rate settings - reduce initial LR to prevent overfitting
        lr0=0.005, # Reduced initial learning rate (was 0.01)
        lrf=0.0005, # Reduced final learning rate (was 0.001)
        cos_lr=True, # Use cosine annealing for learning rate scheduling
        
        # Regularization improvements
        weight_decay=0.0005, # L2 regularization (default is 0.0005)
        dropout=0.1, # Dropout rate for regularization
        
        # Early stopping with more patience to avoid stopping too early
        patience=30, # Increased patience (was 20)
        
        # Data augmentation improvements
        hsv_h=0.015, # Hue augmentation (default 0.015)
        hsv_s=0.7, # Saturation augmentation (default 0.7)
        hsv_v=0.4, # Value augmentation (default 0.4)
        degrees=10.0, # Rotation augmentation (default 0.0)
        translate=0.1, # Translation augmentation (default 0.1)
        scale=0.5, # Scale augmentation (default 0.5)
        shear=2.0, # Shear augmentation (default 0.0)
        perspective=0.0001, # Perspective augmentation (default 0.0)
        flipud=0.5, # Vertical flip probability (default 0.0)
        fliplr=0.5, # Horizontal flip probability (default 0.5)
        mosaic=0.8, # Mosaic augmentation probability (default 1.0)
        mixup=0.1, # Mixup augmentation probability (default 0.0)
        copy_paste=0.1, # Copy-paste augmentation probability (default 0.0)
                
        # Training settings
        device=args.device, # GPU device
        plots=True, # Plot training results
        
        # Validation settings
        val=True, # Enable validation
        
        # Additional settings to reduce overfitting
        close_mosaic=10, # Disable mosaic augmentation in last 10 epochs
        amp=True, # Enable Automatic Mixed Precision
        fraction=1.0, # Use full dataset
        profile=False, # Disable profiling for faster training
        freeze=None, # Don't freeze any layers
        
        # Optimizer settings
        optimizer='AdamW', # Use AdamW optimizer (better for generalization)
        momentum=0.937, # Momentum for SGD
        
        # Loss function improvements
        box=7.5, # Box loss weight
        cls=0.5, # Classification loss weight
        dfl=1.5, # Distribution Focal Loss weight
        
        # Multi-scale training
        rect=False, # Disable rectangular training for more augmentation
        overlap_mask=True, # Enable overlap mask for segmentation
        mask_ratio=4, # Mask downsample ratio
        
        # Workspace settings
        exist_ok=True, # Allow overwriting existing experiment
        pretrained=True, # Use pretrained weights
        verbose=True, # Verbose output
    )

    # Copy the script to the output directory after training starts
    script_copy = output_dir / f"train_yolo11{args.model_size}_{args.target}.py"
    if os.path.exists(__file__):
        shutil.copy(__file__, script_copy)

if __name__ == "__main__":
    main()
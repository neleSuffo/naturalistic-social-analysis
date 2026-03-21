import os
import sys
import torch
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from constants import FaceDetection
from config import FaceConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO model for different detection tasks')
    parser.add_argument('--device', type=str, default='1',
                      help='Device to use (e.g., "0" for GPU, "cpu" for CPU)')
    parser.add_argument('--config', type=str, default=str(FaceDetection.DATA_CONFIG_PATH),
                      help=f'Path to YOLO data config file (default: {FaceDetection.DATA_CONFIG_PATH})')
    return parser.parse_args()

def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set thread limits
    os.environ['OMP_NUM_THREADS'] = '12'
    torch.set_num_threads(12)

    base_output_dir = FaceDetection.OUTPUT_DIR

    # Load the YOLO model - changed default to 'm'
    model_name = FaceConfig.MODEL_NAME
    print(f"Loading model: {model_name}")

    model = YOLO(FaceDetection.FACE_MODEL_PATH)
    
    experiment_name = f"{FaceConfig.MODEL_NAME}_{timestamp}"
    output_dir = base_output_dir / experiment_name

    print(f"Training will be saved to: {output_dir}")
    print("-" * 50)

    # Determine config file path
    if args.config:
        config_file_path = FaceDetection.DATA_CONFIG_PATH.parent / args.config

    # Train the model with improved regularization to reduce overfitting
    model.train(
        data=config_file_path,
        epochs=FaceConfig.NUM_EPOCHS,
        imgsz=FaceConfig.IMG_SIZE,
        batch=FaceConfig.BATCH_SIZE,
        project=str(base_output_dir),
        name=experiment_name,
        augment=True,
        patience=20,
        device=args.device,
        plots=True,
        val=True,
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )

    # Copy the script to the output directory after training starts
    script_copy = output_dir / f"train_{FaceConfig.MODEL_NAME}.py"
    if os.path.exists(__file__):
        shutil.copy(__file__, script_copy)

if __name__ == "__main__":
    main()
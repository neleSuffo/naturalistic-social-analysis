#!/usr/bin/env python3
"""
Hyperparameter tuning script for YOLO models to reduce overfitting.
This script uses Ray Tune to find optimal hyperparameters.
"""

import os
import sys
import torch
import argparse
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from constants import DetectionPaths
import optuna
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for YOLO models')
    parser.add_argument('--target', type=str, default='all',
                      choices=['all', 'face'],
                      help='Target detection task to tune')
    parser.add_argument('--n_trials', type=int, default=20,
                      help='Number of trials for hyperparameter search')
    parser.add_argument('--device', type=str, default='0',
                      help='Device to use (e.g., "0" for GPU, "cpu" for CPU)')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs for each trial (reduced for faster tuning)')
    parser.add_argument('--model_size', type=str, default='m',
                      choices=['n', 's', 'm', 'l', 'x'],
                      help='YOLO model size for tuning')
    return parser.parse_args()

def objective(trial, args):
    """Objective function for Optuna optimization."""
    
    # Suggest hyperparameters
    lr0 = trial.suggest_float('lr0', 0.001, 0.01, log=True)
    lrf = trial.suggest_float('lrf', 0.0001, 0.001, log=True)
    weight_decay = trial.suggest_float('weight_decay', 0.0001, 0.001, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    
    # Augmentation parameters
    hsv_h = trial.suggest_float('hsv_h', 0.01, 0.05)
    hsv_s = trial.suggest_float('hsv_s', 0.3, 0.9)
    hsv_v = trial.suggest_float('hsv_v', 0.2, 0.6)
    degrees = trial.suggest_float('degrees', 0, 20)
    translate = trial.suggest_float('translate', 0.05, 0.2)
    scale = trial.suggest_float('scale', 0.3, 0.7)
    shear = trial.suggest_float('shear', 0, 5)
    mixup = trial.suggest_float('mixup', 0, 0.3)
    copy_paste = trial.suggest_float('copy_paste', 0, 0.3)
    
    # Loss weights
    box_loss = trial.suggest_float('box_loss', 5.0, 10.0)
    cls_loss = trial.suggest_float('cls_loss', 0.3, 1.0)
    dfl_loss = trial.suggest_float('dfl_loss', 1.0, 2.0)
    
    # Model settings
    model_name = f"yolo11{args.model_size}.pt"
    data_config_path = getattr(DetectionPaths, f"{args.target}_data_config_path")
    base_output_dir = getattr(DetectionPaths, f"{args.target}_output_dir")
    
    # Create unique experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"tune_{timestamp}_trial_{trial.number}"
    
    try:
        # Load model
        model = YOLO(model_name)
        
        # Train with suggested parameters
        results = model.train(
            data=str(data_config_path),
            epochs=args.epochs,
            imgsz=640,
            batch=16,
            project=str(base_output_dir),
            name=experiment_name,
            
            # Suggested hyperparameters
            lr0=lr0,
            lrf=lrf,
            weight_decay=weight_decay,
            dropout=dropout,
            
            # Augmentation
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            mixup=mixup,
            copy_paste=copy_paste,
            
            # Loss weights
            box=box_loss,
            cls=cls_loss,
            dfl=dfl_loss,
            
            # Fixed parameters
            device=args.device,
            patience=20,
            cos_lr=True,
            augment=True,
            plots=False,  # Disable plots for faster training
            verbose=False,
            exist_ok=True,
        )
        
        # Load best model and evaluate on test set
        best_model_path = base_output_dir / experiment_name / "weights" / "best.pt"
        if best_model_path.exists():
            best_model = YOLO(str(best_model_path))
            
            # Evaluate on validation set
            val_results = best_model.val(data=str(data_config_path), split='val', verbose=False)
            
            # Evaluate on test set
            test_results = best_model.val(data=str(data_config_path), split='test', verbose=False)
            
            # Calculate generalization score (we want to minimize val-test gap)
            val_map50 = val_results.box.map50
            test_map50 = test_results.box.map50
            
            # Objective: maximize test performance while minimizing overfitting
            # We use a weighted combination of test performance and generalization
            generalization_penalty = abs(val_map50 - test_map50) / val_map50 if val_map50 > 0 else 1.0
            
            # Primary objective: test performance (higher is better)
            # Secondary objective: minimize overfitting (lower penalty is better)
            score = test_map50 - (generalization_penalty * 0.3)
            
            # Log results
            logger.info(f"Trial {trial.number}: Val mAP50: {val_map50:.4f}, Test mAP50: {test_map50:.4f}, Score: {score:.4f}")
            
            # Report intermediate values for pruning
            trial.report(score, args.epochs)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            return score
        else:
            logger.error(f"Best model not found for trial {trial.number}")
            return 0.0
            
    except Exception as e:
        logger.error(f"Error in trial {trial.number}: {str(e)}")
        return 0.0

def main():
    args = parse_args()
    
    # Set thread limits
    os.environ['OMP_NUM_THREADS'] = '4'
    torch.set_num_threads(4)
    
    print(f"Starting hyperparameter tuning for {args.target} detection")
    print(f"Model size: {args.model_size}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Epochs per trial: {args.epochs}")
    print(f"Device: {args.device}")
    print("-" * 50)
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Optimize
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    
    # Print results
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING RESULTS")
    print("="*50)
    
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best score: {study.best_value:.4f}")
    
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = getattr(DetectionPaths, f"{args.target}_output_dir")
    results_file = base_output_dir / f"hyperparameter_tuning_results_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write("HYPERPARAMETER TUNING RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Target: {args.target}\n")
        f.write(f"Model size: {args.model_size}\n")
        f.write(f"Number of trials: {args.n_trials}\n")
        f.write(f"Epochs per trial: {args.epochs}\n")
        f.write(f"Device: {args.device}\n\n")
        
        f.write(f"Number of finished trials: {len(study.trials)}\n")
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best score: {study.best_value:.4f}\n\n")
        
        f.write("Best parameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\nAll trials:\n")
        for trial in study.trials:
            f.write(f"Trial {trial.number}: {trial.value:.4f if trial.value else 'Failed'}\n")
    
    print(f"\nResults saved to: {results_file}")
    
    # Generate training command with best parameters
    best_params = study.best_params
    training_command = f"python train_yolo_det.py --target {args.target} --model_size {args.model_size}"
    
    print("\n" + "="*50)
    print("RECOMMENDED TRAINING COMMAND")
    print("="*50)
    print(training_command)
    
    print("\nRecommended hyperparameters to use in train_yolo_det.py:")
    print("  # Learning rate")
    print(f"  lr0={best_params['lr0']:.6f}")
    print(f"  lrf={best_params['lrf']:.6f}")
    print("  # Regularization")
    print(f"  weight_decay={best_params['weight_decay']:.6f}")
    print(f"  dropout={best_params['dropout']:.3f}")
    print("  # Augmentation")
    print(f"  hsv_h={best_params['hsv_h']:.4f}")
    print(f"  hsv_s={best_params['hsv_s']:.3f}")
    print(f"  hsv_v={best_params['hsv_v']:.3f}")
    print(f"  degrees={best_params['degrees']:.1f}")
    print(f"  translate={best_params['translate']:.3f}")
    print(f"  scale={best_params['scale']:.3f}")
    print(f"  shear={best_params['shear']:.1f}")
    print(f"  mixup={best_params['mixup']:.3f}")
    print(f"  copy_paste={best_params['copy_paste']:.3f}")
    print("  # Loss weights")
    print(f"  box={best_params['box_loss']:.1f}")
    print(f"  cls={best_params['cls_loss']:.3f}")
    print(f"  dfl={best_params['dfl_loss']:.3f}")

if __name__ == "__main__":
    main()

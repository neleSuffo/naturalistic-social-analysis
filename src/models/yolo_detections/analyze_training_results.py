#!/usr/bin/env python3
"""
Script to analyze YOLO training results and provide recommendations for reducing overfitting.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from constants import DetectionPaths

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze YOLO training results')
    parser.add_argument('--experiment_path', type=str, required=True,
                      help='Path to the experiment directory')
    parser.add_argument('--target', type=str, default='all',
                      choices=['all', 'face'],
                      help='Target detection task')
    return parser.parse_args()

def analyze_training_curves(experiment_path):
    """Analyze training curves to detect overfitting."""
    results_file = Path(experiment_path) / "results.csv"
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return None
    
    # Read results
    df = pd.read_csv(results_file)
    print(f"Training completed for {len(df)} epochs")
    
    # Create analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Loss curves
    axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color='blue')
    axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='red')
    axes[0, 0].set_title('Box Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: mAP curves
    axes[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', color='green')
    axes[0, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', color='orange')
    axes[0, 1].set_title('mAP Metrics')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('mAP')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Learning rate
    if 'lr/pg0' in df.columns:
        axes[1, 0].plot(df['epoch'], df['lr/pg0'], label='Learning Rate', color='purple')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Plot 4: Class loss
    if 'train/cls_loss' in df.columns:
        axes[1, 1].plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss', color='blue')
        axes[1, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss', color='red')
        axes[1, 1].set_title('Classification Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(experiment_path) / "training_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training analysis plot saved to: {plot_path}")
    
    return df

def detect_overfitting(df):
    """Detect overfitting patterns in training curves."""
    print("\n" + "="*50)
    print("OVERFITTING ANALYSIS")
    print("="*50)
    
    issues = []
    recommendations = []
    
    # Check if validation loss starts increasing while training loss decreases
    if 'val/box_loss' in df.columns and 'train/box_loss' in df.columns:
        # Find the best validation loss epoch
        best_val_epoch = df['val/box_loss'].idxmin()
        final_epochs = df.tail(20)  # Last 20 epochs
        
        # Check if validation loss increased significantly from best
        best_val_loss = df.loc[best_val_epoch, 'val/box_loss']
        final_val_loss = df['val/box_loss'].iloc[-1]
        
        if final_val_loss > best_val_loss * 1.1:  # 10% increase
            issues.append("Validation loss increased significantly from best")
            recommendations.append("Use early stopping or reduce learning rate")
    
    # Check if training and validation losses diverge
    if 'val/box_loss' in df.columns and 'train/box_loss' in df.columns:
        final_train_loss = df['train/box_loss'].iloc[-1]
        final_val_loss = df['val/box_loss'].iloc[-1]
        
        if final_val_loss > final_train_loss * 1.5:  # Val loss 50% higher than train
            issues.append("Large gap between training and validation loss")
            recommendations.append("Increase regularization (weight_decay, dropout)")
    
    # Check mAP trends
    if 'metrics/mAP50(B)' in df.columns:
        map50_values = df['metrics/mAP50(B)'].values
        # Check if mAP plateaued or decreased
        best_map50_epoch = df['metrics/mAP50(B)'].idxmax()
        best_map50 = df.loc[best_map50_epoch, 'metrics/mAP50(B)']
        final_map50 = df['metrics/mAP50(B)'].iloc[-1]
        
        if final_map50 < best_map50 * 0.95:  # 5% drop from best
            issues.append("mAP50 decreased from best performance")
            recommendations.append("Use early stopping at epoch " + str(best_map50_epoch))
    
    # Print findings
    if issues:
        print("⚠️  OVERFITTING INDICATORS DETECTED:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
        
        print("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("✅ No clear overfitting indicators detected")
    
    return issues, recommendations

def evaluate_model_performance(experiment_path, target):
    """Evaluate model performance on different splits."""
    print("\n" + "="*50)
    print("MODEL PERFORMANCE EVALUATION")
    print("="*50)
    
    best_model_path = Path(experiment_path) / "weights" / "best.pt"
    
    if not best_model_path.exists():
        print(f"Best model not found: {best_model_path}")
        return
    
    print(f"Loading model: {best_model_path}")
    model = YOLO(str(best_model_path))
    
    # Get data config
    data_config_path = getattr(DetectionPaths, f"{target}_data_config_path")
    
    # Evaluate on all splits
    results = {}
    
    for split in ['train', 'val', 'test']:
        print(f"\nEvaluating on {split} set...")
        try:
            split_results = model.val(data=str(data_config_path), split=split)
            results[split] = {
                'mAP50': split_results.box.map50,
                'mAP50-95': split_results.box.map,
                'precision': split_results.box.p.mean(),
                'recall': split_results.box.r.mean(),
            }
        except Exception as e:
            print(f"Error evaluating {split} set: {e}")
            results[split] = None
    
    # Print comparison
    print("\nPERFORMANCE COMPARISON:")
    print("-" * 60)
    print(f"{'Split':<10} {'mAP50':<10} {'mAP50-95':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 60)
    
    for split, result in results.items():
        if result:
            print(f"{split:<10} {result['mAP50']:<10.4f} {result['mAP50-95']:<10.4f} "
                  f"{result['precision']:<10.4f} {result['recall']:<10.4f}")
        else:
            print(f"{split:<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
    
    # Calculate performance drops
    if results['train'] and results['val'] and results['test']:
        train_map50 = results['train']['mAP50']
        val_map50 = results['val']['mAP50']
        test_map50 = results['test']['mAP50']
        
        train_val_drop = (train_map50 - val_map50) / train_map50 * 100
        val_test_drop = (val_map50 - test_map50) / val_map50 * 100
        train_test_drop = (train_map50 - test_map50) / train_map50 * 100
        
        print(f"\nPERFORMANCE DROPS:")
        print(f"Train → Val: {train_val_drop:.1f}%")
        print(f"Val → Test: {val_test_drop:.1f}%")
        print(f"Train → Test: {train_test_drop:.1f}%")
        
        # Provide recommendations based on performance drops
        print("\nRECOMMENDATIONS:")
        if train_test_drop > 20:
            print("🔴 SEVERE OVERFITTING (>20% drop)")
            print("   • Use a smaller model (e.g., YOLOv8m instead of YOLOv8x)")
            print("   • Increase weight_decay to 0.001")
            print("   • Add dropout (0.2-0.3)")
            print("   • Reduce learning rate by 50%")
            print("   • Use early stopping")
        elif train_test_drop > 10:
            print("🟡 MODERATE OVERFITTING (10-20% drop)")
            print("   • Increase data augmentation")
            print("   • Add weight_decay (0.0005)")
            print("   • Use cosine annealing LR schedule")
            print("   • Consider early stopping")
        elif train_test_drop > 5:
            print("🟢 MILD OVERFITTING (5-10% drop)")
            print("   • Fine-tune augmentation parameters")
            print("   • Monitor training more closely")
        else:
            print("✅ GOOD GENERALIZATION (<5% drop)")
    
    return results

def generate_recommendations(issues, performance_results):
    """Generate specific recommendations based on analysis."""
    print("\n" + "="*50)
    print("SPECIFIC RECOMMENDATIONS")
    print("="*50)
    
    recommendations = []
    
    # Based on overfitting analysis
    if issues:
        recommendations.append("IMMEDIATE ACTIONS:")
        recommendations.append("1. Implement early stopping")
        recommendations.append("2. Reduce model complexity")
        recommendations.append("3. Increase regularization")
    
    # Based on performance results
    if performance_results and performance_results.get('train') and performance_results.get('test'):
        train_map50 = performance_results['train']['mAP50']
        test_map50 = performance_results['test']['mAP50']
        drop = (train_map50 - test_map50) / train_map50 * 100
        
        if drop > 15:
            recommendations.append("\nHYPERPARAMETER ADJUSTMENTS:")
            recommendations.append("• learning_rate: 0.003 → 0.001 (reduce by 70%)")
            recommendations.append("• weight_decay: 0.0005 → 0.001 (double)")
            recommendations.append("• dropout: 0.0 → 0.2 (add regularization)")
            recommendations.append("• batch_size: increase if possible")
            recommendations.append("• model_size: consider using 'l' instead of 'x'")
            
            recommendations.append("\nAUGMENTATION ADJUSTMENTS:")
            recommendations.append("• mixup: 0.1 → 0.2 (increase)")
            recommendations.append("• copy_paste: 0.1 → 0.2 (increase)")
            recommendations.append("• degrees: 10 → 15 (more rotation)")
            recommendations.append("• translate: 0.1 → 0.15 (more translation)")
            
            recommendations.append("\nTRAINING STRATEGY:")
            recommendations.append("• Use early stopping with patience=15")
            recommendations.append("• Monitor validation loss closely")
            recommendations.append("• Consider transfer learning approach")
            recommendations.append("• Collect more diverse training data")
    
    for rec in recommendations:
        print(rec)
    
    return recommendations

def main():
    args = parse_args()
    
    experiment_path = Path(args.experiment_path)
    
    if not experiment_path.exists():
        print(f"Experiment path not found: {experiment_path}")
        return
    
    print(f"Analyzing experiment: {experiment_path}")
    print(f"Target: {args.target}")
    
    # Analyze training curves
    df = analyze_training_curves(experiment_path)
    
    if df is not None:
        # Detect overfitting
        issues, _ = detect_overfitting(df)
        
        # Evaluate model performance
        performance_results = evaluate_model_performance(experiment_path, args.target)
        
        # Generate recommendations
        generate_recommendations(issues, performance_results)
        
        # Save analysis report
        report_path = experiment_path / "training_analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write("YOLO TRAINING ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Experiment: {experiment_path}\n")
            f.write(f"Target: {args.target}\n")
            f.write(f"Total epochs: {len(df)}\n\n")
            
            if issues:
                f.write("OVERFITTING ISSUES DETECTED:\n")
                for i, issue in enumerate(issues, 1):
                    f.write(f"{i}. {issue}\n")
                f.write("\n")
            
            if performance_results:
                f.write("PERFORMANCE RESULTS:\n")
                for split, result in performance_results.items():
                    if result:
                        f.write(f"{split}: mAP50={result['mAP50']:.4f}, mAP50-95={result['mAP50-95']:.4f}\n")
                f.write("\n")
        
        print(f"\nAnalysis report saved to: {report_path}")
    
    else:
        print("Could not analyze training curves - results.csv not found")

if __name__ == "__main__":
    main()

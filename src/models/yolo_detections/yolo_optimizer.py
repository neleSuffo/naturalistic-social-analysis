#!/usr/bin/env python3
"""
YOLO Performance Optimization Guide and Implementation
This script provides comprehensive methods to improve YOLO model performance
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import Dict, List, Tuple, Any

class YOLOOptimizer:
    """
    A comprehensive YOLO model optimizer with various performance enhancement techniques
    """
    
    def __init__(self, model_path: str, data_config: str, output_dir: str):
        self.model_path = model_path
        self.data_config = data_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = YOLO(model_path)
        
        # Performance tracking
        self.results = {}
        
    def optimize_hyperparameters(self, generations: int = 30) -> Dict[str, Any]:
        """
        Optimize hyperparameters using evolutionary algorithm
        """
        print(f"Optimizing hyperparameters for {generations} generations...")
        
        results = self.model.tune(
            data=self.data_config,
            epochs=50,  # Reduced epochs for tuning
            iterations=generations,
            optimizer='AdamW',
            plots=True,
            save=True,
            val=True,
            project=str(self.output_dir),
            name="hyperparameter_optimization"
        )
        
        self.results['hyperparameter_optimization'] = results
        return results
    
    def implement_progressive_training(self, total_epochs: int = 300) -> Dict[str, Any]:
        """
        Implement progressive training with different phases
        """
        print("Implementing progressive training...")
        
        # Phase 1: Conservative training (30% of epochs)
        phase1_epochs = int(total_epochs * 0.3)
        print(f"Phase 1: Conservative training ({phase1_epochs} epochs)")
        
        phase1_result = self.model.train(
            data=self.data_config,
            epochs=phase1_epochs,
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            mosaic=0.5,
            mixup=0.0,
            copy_paste=0.0,
            degrees=0.0,
            translate=0.05,
            scale=0.3,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            hsv_h=0.01,
            hsv_s=0.5,
            hsv_v=0.3,
            project=str(self.output_dir),
            name="progressive_phase1",
            exist_ok=True
        )
        
        # Phase 2: Moderate training (40% of epochs)
        phase2_epochs = int(total_epochs * 0.4)
        print(f"Phase 2: Moderate training ({phase2_epochs} epochs)")
        
        phase2_result = self.model.train(
            data=self.data_config,
            epochs=phase2_epochs,
            lr0=0.008,
            lrf=0.008,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=2,
            mosaic=0.8,
            mixup=0.05,
            copy_paste=0.05,
            degrees=5.0,
            translate=0.1,
            scale=0.4,
            shear=2.0,
            perspective=0.0001,
            flipud=0.1,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            project=str(self.output_dir),
            name="progressive_phase2",
            exist_ok=True
        )
        
        # Phase 3: Fine-tuning (30% of epochs)
        phase3_epochs = total_epochs - phase1_epochs - phase2_epochs
        print(f"Phase 3: Fine-tuning ({phase3_epochs} epochs)")
        
        phase3_result = self.model.train(
            data=self.data_config,
            epochs=phase3_epochs,
            lr0=0.001,  # Reduced learning rate
            lrf=0.0001,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=0,
            mosaic=0.3,
            mixup=0.0,
            copy_paste=0.0,
            degrees=0.0,
            translate=0.02,
            scale=0.2,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.3,
            hsv_h=0.005,
            hsv_s=0.3,
            hsv_v=0.2,
            close_mosaic=0,  # Disable mosaic immediately
            project=str(self.output_dir),
            name="progressive_phase3",
            exist_ok=True
        )
        
        results = {
            'phase1': phase1_result,
            'phase2': phase2_result,
            'phase3': phase3_result
        }
        
        self.results['progressive_training'] = results
        return results
    
    def optimize_for_deployment(self, target_device: str = 'cpu') -> Dict[str, Any]:
        """
        Optimize model for deployment (pruning, quantization, etc.)
        """
        print(f"Optimizing model for {target_device} deployment...")
        
        # Export to different formats
        export_results = {}
        
        # Export to ONNX
        try:
            onnx_path = self.model.export(format='onnx', dynamic=True, simplify=True)
            export_results['onnx'] = str(onnx_path)
            print(f"✓ ONNX export successful: {onnx_path}")
        except Exception as e:
            print(f"✗ ONNX export failed: {e}")
        
        # Export to TensorRT (if available)
        try:
            trt_path = self.model.export(format='engine', dynamic=True, simplify=True)
            export_results['tensorrt'] = str(trt_path)
            print(f"✓ TensorRT export successful: {trt_path}")
        except Exception as e:
            print(f"✗ TensorRT export failed: {e}")
        
        # Export to CoreML (for iOS/macOS)
        try:
            coreml_path = self.model.export(format='coreml')
            export_results['coreml'] = str(coreml_path)
            print(f"✓ CoreML export successful: {coreml_path}")
        except Exception as e:
            print(f"✗ CoreML export failed: {e}")
        
        # Export to TFLite (for mobile)
        try:
            tflite_path = self.model.export(format='tflite')
            export_results['tflite'] = str(tflite_path)
            print(f"✓ TFLite export successful: {tflite_path}")
        except Exception as e:
            print(f"✗ TFLite export failed: {e}")
        
        self.results['deployment_optimization'] = export_results
        return export_results
    
    def implement_ensemble_training(self, model_sizes: List[str] = ['s', 'm', 'l']) -> Dict[str, Any]:
        """
        Train multiple models for ensemble
        """
        print(f"Training ensemble with model sizes: {model_sizes}")
        
        ensemble_results = {}
        
        for size in model_sizes:
            print(f"Training YOLO11{size} model...")
            
            model = YOLO(f"yolo11{size}.pt")
            
            # Adjust training parameters based on model size
            if size in ['s', 'n']:
                lr0, batch_size = 0.01, 32
            elif size in ['m']:
                lr0, batch_size = 0.008, 16
            else:  # l, x
                lr0, batch_size = 0.005, 8
            
            result = model.train(
                data=self.data_config,
                epochs=200,
                lr0=lr0,
                batch=batch_size,
                imgsz=640,
                patience=50,
                project=str(self.output_dir),
                name=f"ensemble_{size}",
                exist_ok=True
            )
            
            ensemble_results[size] = result
            print(f"YOLO11{size} training completed")
        
        self.results['ensemble_training'] = ensemble_results
        return ensemble_results
    
    def implement_multiscale_training(self, scales: List[int] = [512, 640, 768]) -> Dict[str, Any]:
        """
        Implement multi-scale training for better generalization
        """
        print(f"Implementing multi-scale training with scales: {scales}")
        
        multiscale_results = {}
        
        for scale in scales:
            print(f"Training with image size: {scale}")
            
            # Adjust batch size based on image size
            if scale <= 512:
                batch_size = 32
            elif scale <= 640:
                batch_size = 16
            else:
                batch_size = 8
            
            result = self.model.train(
                data=self.data_config,
                epochs=100,
                imgsz=scale,
                batch=batch_size,
                lr0=0.008,
                rect=False,  # Disable rectangular training
                mosaic=1.0,
                mixup=0.1,
                project=str(self.output_dir),
                name=f"multiscale_{scale}",
                exist_ok=True
            )
            
            multiscale_results[scale] = result
            print(f"Training with scale {scale} completed")
        
        self.results['multiscale_training'] = multiscale_results
        return multiscale_results
    
    def validate_with_tta(self, augment_configs: List[Dict] = None) -> Dict[str, Any]:
        """
        Validate model with Test-Time Augmentation
        """
        if augment_configs is None:
            augment_configs = [
                {'conf': 0.001, 'iou': 0.7, 'augment': True},
                {'conf': 0.005, 'iou': 0.6, 'augment': True},
                {'conf': 0.01, 'iou': 0.8, 'augment': True},
            ]
        
        print("Validating with Test-Time Augmentation...")
        
        tta_results = {}
        
        for i, config in enumerate(augment_configs):
            print(f"TTA configuration {i+1}: {config}")
            
            result = self.model.val(
                data=self.data_config,
                split='test',
                save_json=True,
                project=str(self.output_dir),
                name=f"tta_config_{i+1}",
                **config
            )
            
            tta_results[f'config_{i+1}'] = {
                'config': config,
                'mAP50': result.results_dict.get('metrics/mAP50(B)', 0),
                'mAP50_95': result.results_dict.get('metrics/mAP50-95(B)', 0),
                'precision': result.results_dict.get('metrics/precision(B)', 0),
                'recall': result.results_dict.get('metrics/recall(B)', 0)
            }
            
            print(f"TTA Config {i+1} - mAP50: {tta_results[f'config_{i+1}']['mAP50']:.4f}")
        
        self.results['tta_validation'] = tta_results
        return tta_results
    
    def analyze_model_performance(self) -> Dict[str, Any]:
        """
        Analyze model performance and provide recommendations
        """
        print("Analyzing model performance...")
        
        analysis = {
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.model.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.model.parameters() if p.requires_grad),
                'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.model.parameters()) / (1024 * 1024)
            },
            'recommendations': [],
            'best_techniques': {}
        }
        
        # Analyze results and provide recommendations
        if 'tta_validation' in self.results:
            best_tta = max(self.results['tta_validation'].items(), key=lambda x: x[1]['mAP50'])
            analysis['best_techniques']['tta'] = best_tta
            analysis['recommendations'].append(f"Best TTA configuration: {best_tta[0]} with mAP50: {best_tta[1]['mAP50']:.4f}")
        
        if 'ensemble_training' in self.results:
            analysis['recommendations'].append("Consider ensemble averaging for better performance")
        
        if 'multiscale_training' in self.results:
            best_scale = max(self.results['multiscale_training'].items(), key=lambda x: x[1].results_dict.get('metrics/mAP50(B)', 0))
            analysis['best_techniques']['multiscale'] = best_scale
            analysis['recommendations'].append(f"Best training scale: {best_scale[0]} pixels")
        
        # Model size recommendations
        model_params = analysis['model_info']['total_parameters']
        if model_params > 50_000_000:
            analysis['recommendations'].append("Model is large - consider pruning or knowledge distillation for deployment")
        elif model_params < 10_000_000:
            analysis['recommendations'].append("Model is small - consider using a larger model for potentially better performance")
        
        self.results['performance_analysis'] = analysis
        return analysis
    
    def save_optimization_report(self, filename: str = "optimization_report.json"):
        """
        Save comprehensive optimization report
        """
        report_path = self.output_dir / filename
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"Optimization report saved to: {report_path}")
        return report_path
    
    def run_complete_optimization(self):
        """
        Run the complete optimization pipeline
        """
        print("Starting complete YOLO optimization pipeline...")
        
        # 1. Hyperparameter optimization
        self.optimize_hyperparameters(generations=20)
        
        # 2. Progressive training
        self.implement_progressive_training(total_epochs=300)
        
        # 3. Multi-scale training
        self.implement_multiscale_training()
        
        # 4. Test-time augmentation validation
        self.validate_with_tta()
        
        # 5. Performance analysis
        self.analyze_model_performance()
        
        # 6. Deployment optimization
        self.optimize_for_deployment()
        
        # 7. Save report
        self.save_optimization_report()
        
        print("Complete optimization pipeline finished!")
        return self.results

# Example usage
if __name__ == "__main__":
    # Configuration
    model_path = "yolo11m.pt"
    data_config = "path/to/your/data.yaml"
    output_dir = "/home/nele_pauline_suffo/yolo_optimization_results"
    
    # Initialize optimizer
    optimizer = YOLOOptimizer(model_path, data_config, output_dir)
    
    # Run complete optimization
    results = optimizer.run_complete_optimization()
    
    print("Optimization completed!")
    print(f"Results saved to: {output_dir}")
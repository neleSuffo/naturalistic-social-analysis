import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataloader import default_collate
from torchvision import transforms, models
from PIL import Image
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc  # For garbage collection
import copy  # For deep copying model state
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Import for mixed precision training
try:
    from torch.amp import autocast, GradScaler
    MIXED_PRECISION_AVAILABLE = True
except ImportError:
    try:
        # Fallback to older API
        from torch.cuda.amp import autocast, GradScaler
        MIXED_PRECISION_AVAILABLE = True
    except ImportError:
        MIXED_PRECISION_AVAILABLE = False
        print("Mixed precision training not available in this PyTorch version")

# Set memory management environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Clear GPU cache at the start
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# --- 1. Configuration and Hyperparameters ---
class Config:
    IMAGE_SIZE = 288  # Further increased from 256 to 288 for higher GPU memory usage
    BATCH_SIZE = 12  # Reduced from 16 to 12 to ensure stability
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Memory optimization settings
    NUM_WORKERS = 6  # Increased from 4 to 6 for better data loading throughput
    PIN_MEMORY = True  # Enable for faster GPU transfers
    GRADIENT_ACCUMULATION_STEPS = 2  # Reduced from 3 to 2 since we increased batch size significantly
    
    # Enable mixed precision training
    USE_MIXED_PRECISION = True
    
    # GPU utilization settings
    USE_GRADIENT_CHECKPOINTING = True  # Save memory to allow larger batches
    PREFETCH_FACTOR = 6  # Increased from 4 to 6 - number of batches to prefetch
    PERSISTENT_WORKERS = True  # Keep data loading workers alive between epochs
    USE_TORCH_COMPILE = True  # Use torch.compile for better performance (PyTorch 2.0+)
    
    # Memory clearing frequency
    MEMORY_CLEAR_FREQUENCY = 50  # Further increased from 30 to 50 to allow maximum GPU utilization
    
    # Debugging settings
    DEBUG_MODE = True  # Enable debugging features like overfitting test
    
    # Dataset validation
    VALIDATE_IMAGES = True  # Whether to check if all images exist during dataset creation
    
    # Model configuration
    BACKBONE_NAME = 'resnet50'  # Backbone architecture
    USE_PRETRAINED_BACKBONE = True  # Whether to use pretrained weights
    
    # Output settings
    SAVE_PLOTS = True  # Whether to save training plots
    SAVE_RESULTS = True  # Whether to save detailed results to JSON
    SAVE_SCRIPT_COPY = True  # Whether to save a copy of the training script
    BASE_OUTPUT_DIR = "/home/nele_pauline_suffo/outputs/cnn_classifications"  # Base directory for all experiments
    
    # This will be set dynamically during runtime
    OUTPUT_DIR = None  # Will be set to timestamped folder
    
    # Define labels for the classification task
    LABELS = [
        "adult_person_present",
        "child_person_present",
        "adult_face_present",
        "child_face_present",
        "object_interaction",
    ]
    NUM_LABELS = len(LABELS)

    DATA_ROOT = "/home/nele_pauline_suffo/ProcessedData/quantex_videos_processed"
    TRAIN_ANNOTATIONS_FILE = "/home/nele_pauline_suffo/ProcessedData/cnn_input/train_annotations.csv"
    VAL_ANNOTATIONS_FILE = "/home/nele_pauline_suffo/ProcessedData/cnn_input/val_annotations.csv"
    TEST_ANNOTATIONS_FILE = "/home/nele_pauline_suffo/ProcessedData/cnn_input/test_annotations.csv"
    
    # Early Stopping Parameters
    EARLY_STOPPING_PATIENCE = 10 # Number of epochs to wait for improvement
    EARLY_STOPPING_MIN_DELTA = 0.001 # Minimum change to qualify as an improvement
    MONITOR_METRIC = 'macro_f1' # Metric to monitor for early stopping
    MONITOR_MODE = 'max' # 'max' for F1-score (we want to maximize it)
    
cfg = Config()
print(f"Using device: {cfg.DEVICE}")

# --- Early Stopping Class ---
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, mode='min', verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.best_score = None
        self.epochs_no_improvement = 0
        self.early_stop = False
        self.best_model_state = None

        if self.mode == 'min':
            self.val_score_sign = 1
            self.best_score = float('inf')
        elif self.mode == 'max':
            self.val_score_sign = -1
            self.best_score = float('-inf')
        else:
            raise ValueError("mode must be 'min' or 'max'")

    def __call__(self, current_score, model):
        # Convert current_score to a comparable value based on mode
        score_to_compare = current_score * self.val_score_sign

        if self.best_score is None:
            self.best_score = current_score
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif score_to_compare < (self.best_score * self.val_score_sign - self.min_delta):
            # Improvement detected
            if self.verbose:
                print(f"Validation score improved ({self.best_score:.4f} -> {current_score:.4f}). Saving model.")
            self.best_score = current_score
            self.epochs_no_improvement = 0
            self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            self.epochs_no_improvement += 1
            if self.verbose:
                print(f"Validation score did not improve. Patience: {self.epochs_no_improvement}/{self.patience}")
            if self.epochs_no_improvement >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")
        
# --- 2. Custom Dataset ---
class EgocentricFrameDataset(Dataset):
    def __init__(self, annotations_df, data_root, transform=None, validate_images=True):
        self.data_root = data_root
        self.transform = transform
        
        # Ensure 'file_name' or similar column exists in your DataFrame
        if 'file_name' not in annotations_df.columns:
            raise ValueError("Annotations DataFrame must contain a 'file_name' column.")
        
        # Ensure all labels are present in the DataFrame
        for label in cfg.LABELS:
            if label not in annotations_df.columns:
                raise ValueError(f"Label column '{label}' not found in annotations DataFrame.")

        # Optionally validate that all images exist
        if validate_images:
            print("Validating image paths...")
            valid_indices = []
            missing_count = 0
            
            for idx, row in annotations_df.iterrows():
                file_name = row['file_name']
                base_name = os.path.basename(file_name)
                folder_name = '_'.join(base_name.split('_')[:-1])
                img_path = os.path.join(data_root, folder_name, file_name)
                
                if os.path.exists(img_path):
                    try:
                        # Quick validation - try to open the image
                        with Image.open(img_path) as img:
                            img.verify()
                        valid_indices.append(idx)
                    except (OSError, IOError):
                        print(f"Warning: Corrupted image {img_path}")
                        missing_count += 1
                else:
                    print(f"Warning: Missing image {img_path}")
                    missing_count += 1
            
            # Filter to only valid images
            self.annotations = annotations_df.iloc[valid_indices].reset_index(drop=True)
            
            print(f"Dataset validation complete:")
            print(f"  Original annotations: {len(annotations_df)}")
            print(f"  Valid images found: {len(self.annotations)}")
            print(f"  Missing/corrupted images: {missing_count}")
            
            if len(self.annotations) == 0:
                raise ValueError("No valid images found in the dataset!")
        else:
            self.annotations = annotations_df

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        frame_info = self.annotations.iloc[idx]
        file_name = frame_info['file_name']
        
        # Extract folder name from file name
        base_name = os.path.basename(file_name)
        folder_name = '_'.join(base_name.split('_')[:-1])  # remove the last part (e.g., '007080.jpg')
        
        img_path = os.path.join(self.data_root, folder_name, file_name)

        try:
            # Try to load the image
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            
            image = Image.open(img_path).convert("RGB")
            
            # Verify image is valid
            image.verify()
            # Reopen since verify() closes the file
            image = Image.open(img_path).convert("RGB")
            
        except (FileNotFoundError, OSError, IOError) as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            # Create a black placeholder image of the correct size
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
    
        if self.transform:
            image = self.transform(image)

        # Labels as a torch tensor (multi-hot encoding for multi-label classification)
        labels = torch.FloatTensor([frame_info[label] for label in cfg.LABELS])
        
        return image, labels

# --- 3. Model Architecture ---
class MultiLabelClassifier(nn.Module):
    def __init__(self, num_labels, backbone_name='resnet50', pretrained=True):
        super(MultiLabelClassifier, self).__init__()
        
        # Load pre-trained backbone
        if backbone_name == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            # Remove the original classification head (avgpool and fc layers)
            self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-2]) 
            self.num_features = self.backbone.fc.in_features # This will be 2048 for ResNet50
        elif backbone_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            # EfficientNet features are in backbone.features
            self.feature_extractor = self.backbone.features
            self.num_features = self.backbone.classifier[1].in_features # This is the input feature size for the classifier
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Global Average Pooling to flatten spatial features for the dense layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Shared feature extractor with maximum capacity for high GPU utilization
        self.shared_features = nn.Sequential(
            nn.Linear(self.num_features, 1536),  # Increased from 1024 to 1536
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1536, 1024),  # Increased from 512 to 1024
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),  # Keep this layer
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),  # Keep this layer
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Simpler multi-head classifier with shared features
        self.classification_heads = nn.ModuleDict()
        for i, label_name in enumerate(cfg.LABELS):
            self.classification_heads[label_name] = nn.Linear(256, 1)  # Remove sigmoid - will use BCEWithLogitsLoss

    def forward(self, x):
        # Use gradient checkpointing to save memory while allowing larger batches
        if self.training:
            # Enable gradient checkpointing for the feature extractor during training
            features = torch.utils.checkpoint.checkpoint(self.feature_extractor, x)
        else:
            features = self.feature_extractor(x)
            
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        
        # Process through shared feature layer
        shared_feat = self.shared_features(features)
        
        # Apply each classification head
        outputs = {}
        for label_name in cfg.LABELS:
            outputs[label_name] = self.classification_heads[label_name](shared_feat).squeeze(1)

        return outputs

# --- 4. Training Function ---
def train_model(model, dataloader, optimizer, criterion, epoch, scaler=None, scheduler=None):
    model.train()
    total_loss = 0
    correct_predictions = {label: 0 for label in cfg.LABELS}
    total_samples = {label: 0 for label in cfg.LABELS}
    
    # Initialize gradient accumulation
    accumulated_loss = 0
    
    import time
    epoch_start_time = time.time()

    total_batches = len(dataloader)
    print(f"Starting epoch {epoch} with {total_batches} batches...")
    print(f"DataLoader length: {total_batches}")
    print(f"Batch size: {cfg.BATCH_SIZE}")
    print(f"Dataset size: {len(dataloader.dataset)}")
    
    print("About to start iterating through DataLoader...")
    
    # Use tqdm for progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training", unit="batch")
    
    # Track optimizer step count for scheduler
    optimizer_step_count = 0
    
    for batch_idx, batch in enumerate(pbar):
        if batch_idx == 0:
            print(f"Successfully started iteration! Processing first batch...")
            
        # Skip corrupted batches
        if batch is None:
            print(f"Warning: Skipping corrupted batch {batch_idx}")
            continue
            
        images, labels_tensor = batch
        images = images.to(cfg.DEVICE, non_blocking=True)
        labels_tensor = labels_tensor.to(cfg.DEVICE, non_blocking=True)

        # Use mixed precision if available
        if cfg.USE_MIXED_PRECISION and scaler is not None:
            with autocast('cuda'):
                outputs = model(images)
                
                # Calculate loss for each head
                loss = 0
                for i, label_name in enumerate(cfg.LABELS):
                    target_labels = labels_tensor[:, i]
                    prediction_output = outputs[label_name]
                    label_loss = criterion(prediction_output, target_labels)
                    loss += label_loss
                    
                    # Debug: Print loss details for first batch of first epoch
                    if epoch == 1 and batch_idx == 0:
                        print(f"      {label_name} loss: {label_loss.item():.4f}, target mean: {target_labels.mean().item():.3f}, pred mean: {torch.sigmoid(prediction_output).mean().item():.3f}")
                
                # Scale loss for gradient accumulation
                loss = loss / cfg.GRADIENT_ACCUMULATION_STEPS
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
            
        else:
            outputs = model(images)
            
            # Calculate loss for each head
            loss = 0
            for i, label_name in enumerate(cfg.LABELS):
                target_labels = labels_tensor[:, i]
                prediction_output = outputs[label_name]
                label_loss = criterion(prediction_output, target_labels)
                loss += label_loss
                
                # Debug: Print loss details for first batch of first epoch (non-mixed precision)
                if epoch == 1 and batch_idx == 0:
                    print(f"      {label_name} loss (no AMP): {label_loss.item():.4f}, target mean: {target_labels.mean().item():.3f}, pred mean: {torch.sigmoid(prediction_output).mean().item():.3f}")
            
            # Scale loss for gradient accumulation
            loss = loss / cfg.GRADIENT_ACCUMULATION_STEPS
            loss.backward()
        
        accumulated_loss += loss.item()
        
        # Calculate accuracy metrics
        for i, label_name in enumerate(cfg.LABELS):
            target_labels = labels_tensor[:, i]
            prediction_output = outputs[label_name]
            # Apply sigmoid for evaluation since we removed it from the model
            sigmoid_probs = torch.sigmoid(prediction_output)
            preds = (sigmoid_probs > 0.5).float()
            correct_predictions[label_name] += (preds == target_labels).sum().item()
            total_samples[label_name] += target_labels.size(0)
            
            # Debug: Print prediction statistics for first batch of first epoch
            if epoch == 1 and batch_idx == 0:
                pos_preds = preds.sum().item()
                pos_targets = target_labels.sum().item()
                avg_prob = sigmoid_probs.mean().item()
                max_prob = sigmoid_probs.max().item()
                min_prob = sigmoid_probs.min().item()
                print(f"    First batch {label_name}: pos_preds={pos_preds}/{len(preds)}, pos_targets={pos_targets}/{len(target_labels)}, avg_prob={avg_prob:.3f}, max_prob={max_prob:.3f}, min_prob={min_prob:.3f}")

        # Perform gradient accumulation step
        if (batch_idx + 1) % cfg.GRADIENT_ACCUMULATION_STEPS == 0:
            # Gradient clipping to prevent exploding gradients
            if cfg.USE_MIXED_PRECISION and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Step the OneCycleLR scheduler after each optimizer step
            if scheduler:
                scheduler.step()
                optimizer_step_count += 1
            
            optimizer.zero_grad()
            total_loss += accumulated_loss
            accumulated_loss = 0

        # Update progress bar with current loss and GPU memory info
        current_loss = loss.item() * cfg.GRADIENT_ACCUMULATION_STEPS
        postfix_dict = {'batch_loss': f'{current_loss:.4f}'}
        
        if torch.cuda.is_available() and batch_idx % 10 == 0:  # Update GPU info every 10 batches
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            postfix_dict['GPU_GB'] = f'{gpu_memory:.1f}'
        
        # Add current learning rate to progress bar
        if scheduler and optimizer_step_count > 0:
            current_lr = scheduler.get_last_lr()[0]
            postfix_dict['lr'] = f'{current_lr:.2e}'
        
        pbar.set_postfix(postfix_dict)
        
        # Aggressive memory management
        if batch_idx % cfg.MEMORY_CLEAR_FREQUENCY == 0:
            torch.cuda.empty_cache()
            gc.collect()
            
        # Print detailed progress every 200 batches (less frequent to avoid spam)
        if batch_idx % 200 == 0 and batch_idx > 0:
            progress_pct = (batch_idx / total_batches) * 100
            if torch.cuda.is_available():
                print(f"\nEpoch {epoch} - Progress: {progress_pct:.1f}% ({batch_idx}/{total_batches})")
                # Print gradient statistics
                total_grad_norm = 0
                for param in model.parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                print(f"  Gradient norm: {total_grad_norm:.4f}")
            
        # Delete variables to free memory immediately
        del images, labels_tensor, outputs, loss
        
    pbar.close()  # Close the progress bar
        
    # Handle remaining gradients if batch size doesn't divide evenly
    if accumulated_loss > 0:
        if cfg.USE_MIXED_PRECISION and scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Step the scheduler for remaining gradients
        if scheduler:
            scheduler.step()
            
        optimizer.zero_grad()
        total_loss += accumulated_loss

    # Fix the average loss calculation
    effective_batches = max(1, len(dataloader) // cfg.GRADIENT_ACCUMULATION_STEPS)
    avg_loss = total_loss / effective_batches
    
    accuracies = {label: (correct_predictions[label] / total_samples[label] if total_samples[label] > 0 else 0) * 100 for label in correct_predictions}
    
    print(f"Epoch {epoch} Average Training Loss: {avg_loss:.4f}")
    print(f"  Total optimizer steps: {optimizer_step_count}")
    for label, acc in accuracies.items():
        print(f"  {label} Training Accuracy: {acc:.2f}%")
        
    # Note: OneCycleLR scheduler is stepped during training, not here
    
    # Final memory cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    return avg_loss

# --- 5. Evaluation Function ---
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds = {label: [] for label in cfg.LABELS}
    all_targets = {label: [] for label in cfg.LABELS}
    all_probs = {label: [] for label in cfg.LABELS}  # Track probabilities for debugging
    
    total_batches = len(dataloader)
    print(f"Starting evaluation with {total_batches} batches...")

    with torch.no_grad():
        # Use tqdm for evaluation progress bar
        pbar = tqdm(dataloader, desc="Evaluating", unit="batch")
        
        for batch_idx, batch in enumerate(pbar):
            # Skip corrupted batches
            if batch is None:
                print(f"Warning: Skipping corrupted batch {batch_idx} during evaluation")
                continue
                
            images, labels_tensor = batch
            images = images.to(cfg.DEVICE, non_blocking=True)
            labels_tensor = labels_tensor.to(cfg.DEVICE, non_blocking=True)

            # Use mixed precision for evaluation too
            if cfg.USE_MIXED_PRECISION:
                with autocast('cuda'):
                    outputs = model(images)
                    
                    loss = 0
                    for i, label_name in enumerate(cfg.LABELS):
                        target_labels = labels_tensor[:, i]
                        prediction_output = outputs[label_name]
                        loss += criterion(prediction_output, target_labels)
            else:
                outputs = model(images)
                
                loss = 0
                for i, label_name in enumerate(cfg.LABELS):
                    target_labels = labels_tensor[:, i]
                    prediction_output = outputs[label_name]
                    loss += criterion(prediction_output, target_labels)
            
            # Collect predictions and targets
            for i, label_name in enumerate(cfg.LABELS):
                target_labels = labels_tensor[:, i]
                prediction_output = outputs[label_name]
                
                # Apply sigmoid for evaluation since we removed it from the model
                sigmoid_probs = torch.sigmoid(prediction_output)
                preds = (sigmoid_probs > 0.5).float().cpu().numpy()
                targets = target_labels.cpu().numpy()
                probs = sigmoid_probs.cpu().numpy()
                
                all_preds[label_name].extend(preds)
                all_targets[label_name].extend(targets)
                all_probs[label_name].extend(probs)
                
                # Debug: Print some prediction statistics for the first few batches
                if batch_idx < 3:  # Only first 3 batches to avoid spam
                    pos_preds = np.sum(preds)
                    pos_targets = np.sum(targets)
                    avg_prob = float(sigmoid_probs.mean())
                    max_prob = float(sigmoid_probs.max())
                    min_prob = float(sigmoid_probs.min())
                    print(f"    Eval batch {batch_idx} {label_name}: pos_preds={pos_preds}/{len(preds)}, pos_targets={pos_targets}/{len(targets)}, avg_prob={avg_prob:.3f}, max_prob={max_prob:.3f}, min_prob={min_prob:.3f}")

            total_loss += loss.item()
            
            # Update progress bar with current loss
            pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
            
            # Extremely infrequent memory clearing during evaluation to maximize GPU utilization
            if batch_idx % (cfg.MEMORY_CLEAR_FREQUENCY * 5) == 0:  # Increased multiplier from 3 to 5
                torch.cuda.empty_cache()
                gc.collect()
                
            # Delete variables to free memory
            del images, labels_tensor, outputs, loss
            
        pbar.close()  # Close the progress bar
        
        # Clear GPU cache after evaluation
        torch.cuda.empty_cache()
        gc.collect()

    avg_loss = total_loss / len(dataloader)

    # Debug: Print detailed prediction statistics before calculating metrics
    print("\n=== Evaluation Prediction Statistics ===")
    for label_name in cfg.LABELS:
        if len(all_targets[label_name]) > 0:
            total_samples = len(all_targets[label_name])
            positive_targets = np.sum(all_targets[label_name])
            positive_predictions = np.sum(all_preds[label_name])
            avg_prob = np.mean(all_probs[label_name])
            max_prob = np.max(all_probs[label_name])
            min_prob = np.min(all_probs[label_name])
            
            print(f"{label_name}:")
            print(f"  Total samples: {total_samples}")
            print(f"  Positive targets: {positive_targets} ({positive_targets/total_samples*100:.1f}%)")
            print(f"  Positive predictions: {positive_predictions} ({positive_predictions/total_samples*100:.1f}%)")
            print(f"  Avg probability: {avg_prob:.3f}, Max: {max_prob:.3f}, Min: {min_prob:.3f}")
            
            # Check for pathological cases
            if avg_prob < 0.01:
                print(f"  WARNING: Very low average probability for {label_name}! Model might be predicting all zeros.")
            elif avg_prob > 0.99:
                print(f"  WARNING: Very high average probability for {label_name}! Model might be predicting all ones.")
            elif max_prob - min_prob < 0.01:
                print(f"  WARNING: Very small probability range for {label_name}! Model might not be learning.")
    print("=" * 40)

    # Calculate metrics for each label
    per_label_metrics = {}
    for label_name in cfg.LABELS:
        # Avoid division by zero if a class has no instances or no predictions
        if len(all_targets[label_name]) > 0:
            # Try different thresholds to see if the issue is with the threshold
            targets = np.array(all_targets[label_name])
            preds = np.array(all_preds[label_name])
            
            # Check if all predictions are 0
            if np.sum(preds) == 0:
                print(f"Warning: All predictions for {label_name} are 0! This might indicate a threshold issue.")
                
                # Try a lower threshold (0.3 instead of 0.5)
                probs = np.array(all_probs[label_name])
                preds_low_thresh = (probs > 0.3).astype(float)
                pos_preds_low = np.sum(preds_low_thresh)
                print(f"  With threshold 0.3: {pos_preds_low} positive predictions")
                
            per_label_metrics[label_name] = {
                'accuracy': accuracy_score(targets, preds),
                'precision': precision_score(targets, preds, zero_division=0),
                'recall': recall_score(targets, preds, zero_division=0),
                'f1': f1_score(targets, preds, zero_division=0)
            }
        else:
            # If a label has no instances in the current split, its metrics are undefined.
            # We can set them to 0 or NaN, here setting to 0 for simplicity in display.
            per_label_metrics[label_name] = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}


    # Calculate Micro and Macro F1-scores for the entire multi-label problem
    all_preds_flat = np.concatenate([all_preds[label] for label in cfg.LABELS])
    all_targets_flat = np.concatenate([all_targets[label] for label in cfg.LABELS])

    micro_f1 = f1_score(all_targets_flat, all_preds_flat, average='micro', zero_division=0) 
    macro_f1 = f1_score(all_targets_flat, all_preds_flat, average='macro', zero_division=0) 

    # Print general validation metrics
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"  Overall Micro F1-score: {micro_f1:.4f}")
    print(f"  Overall Macro F1-score: {macro_f1:.4f}")

    # Return all calculated metrics
    return avg_loss, micro_f1, macro_f1, per_label_metrics

# --- Overfitting Test Function ---
def test_overfitting_capability(model, train_loader, criterion, optimizer, scaler=None):
    """
    Test if the model can overfit on a small batch to verify the training pipeline.
    This is a debugging function to ensure gradients are flowing properly.
    """
    print("\n--- Testing Model's Ability to Overfit on Small Sample ---")
    
    # Get a single batch
    for batch in train_loader:
        if batch is not None:
            images, labels_tensor = batch
            break
    else:
        print("No valid batch found for overfitting test")
        return False
    
    # Take only a small subset (e.g., first 4 samples)
    small_batch_size = min(4, len(images))
    images = images[:small_batch_size].to(cfg.DEVICE)
    labels_tensor = labels_tensor[:small_batch_size].to(cfg.DEVICE)
    
    print(f"Testing overfitting on {small_batch_size} samples...")
    
    model.train()
    initial_loss = None
    
    # Train for a few iterations
    for iteration in range(20):
        optimizer.zero_grad()
        
        if cfg.USE_MIXED_PRECISION and scaler is not None:
            with autocast('cuda'):
                outputs = model(images)
                
                loss = 0
                for i, label_name in enumerate(cfg.LABELS):
                    target_labels = labels_tensor[:, i]
                    prediction_output = outputs[label_name]
                    label_loss = criterion(prediction_output, target_labels)
                    loss += label_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            
            loss = 0
            for i, label_name in enumerate(cfg.LABELS):
                target_labels = labels_tensor[:, i]
                prediction_output = outputs[label_name]
                label_loss = criterion(prediction_output, target_labels)
                loss += label_loss
            
            loss.backward()
            optimizer.step()
        
        if iteration == 0:
            initial_loss = loss.item()
        
        if (iteration + 1) % 5 == 0:
            print(f"  Iteration {iteration + 1}: Loss = {loss.item():.4f}")
    
    final_loss = loss.item()
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Loss reduction: {loss_reduction:.1f}%")
    
    # Check if model can overfit (loss should decrease significantly)
    if loss_reduction > 10:  # At least 10% reduction
        print("✓ Model can overfit successfully - training pipeline is working")
        return True
    else:
        print("✗ Model failed to overfit - there might be an issue with the training pipeline")
        return False

# --- Custom Collate Function ---
def safe_collate_fn(batch):
    """
    Custom collate function that filters out None values and handles batch errors gracefully.
    """
    # Filter out None values (corrupted/missing images)
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        # If entire batch is corrupted, return None - this will be handled in training loop
        return None
    
    # Use default collate function for valid items
    return default_collate(batch)

# --- Experiment Setup Function ---
def setup_experiment_directory():
    """
    Create a timestamped experiment directory and copy the current script.
    Returns the path to the created directory.
    """
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create experiment directory
    experiment_name = f"experiment_{timestamp}"
    experiment_dir = os.path.join(cfg.BASE_OUTPUT_DIR, experiment_name)
    
    # Create all necessary subdirectories
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)
    
    print(f"Created experiment directory: {experiment_dir}")
    
    # Copy the current script to the experiment directory
    if cfg.SAVE_SCRIPT_COPY:
        import shutil
        script_path = __file__  # Current script path
        script_name = os.path.basename(script_path)
        script_copy_path = os.path.join(experiment_dir, f"train_script_{timestamp}.py")
        
        try:
            shutil.copy2(script_path, script_copy_path)
            print(f"Script copied to: {script_copy_path}")
        except Exception as e:
            print(f"Warning: Could not copy script: {e}")
    
    # Create experiment info file
    experiment_info = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "datetime": datetime.now().isoformat(),
        "script_name": os.path.basename(__file__),
        "working_directory": os.getcwd(),
        "config_snapshot": {
            "batch_size": cfg.BATCH_SIZE,
            "learning_rate": cfg.LEARNING_RATE,
            "num_epochs": cfg.NUM_EPOCHS,
            "image_size": cfg.IMAGE_SIZE,
            "gradient_accumulation_steps": cfg.GRADIENT_ACCUMULATION_STEPS,
            "use_mixed_precision": cfg.USE_MIXED_PRECISION,
            "monitor_metric": cfg.MONITOR_METRIC,
            "early_stopping_patience": cfg.EARLY_STOPPING_PATIENCE,
            "data_root": cfg.DATA_ROOT,
            "train_annotations_file": cfg.TRAIN_ANNOTATIONS_FILE,
            "val_annotations_file": cfg.VAL_ANNOTATIONS_FILE,
            "test_annotations_file": cfg.TEST_ANNOTATIONS_FILE
        }
    }
    
    # Save experiment info
    info_path = os.path.join(experiment_dir, "experiment_info.json")
    with open(info_path, 'w') as f:
        json.dump(experiment_info, f, indent=2)
    
    print(f"Experiment info saved to: {info_path}")
    
    return experiment_dir

# --- Logging Setup Function ---
def setup_logging(experiment_dir):
    """
    Setup logging to save console output to a file in the experiment directory.
    """
    import sys
    
    log_file_path = os.path.join(experiment_dir, "logs", "training_log.txt")
    
    # Create a custom logger that writes to both console and file
    class TeeLogger:
        def __init__(self, file_path):
            self.terminal = sys.stdout
            self.log_file = open(file_path, 'w')
        
        def write(self, message):
            self.terminal.write(message)
            self.log_file.write(message)
            self.log_file.flush()
        
        def flush(self):
            self.terminal.flush()
            self.log_file.flush()
        
        def close(self):
            self.log_file.close()
    
    # Redirect stdout to our custom logger
    tee = TeeLogger(log_file_path)
    sys.stdout = tee
    
    print(f"Logging setup complete. Console output will be saved to: {log_file_path}")
    
    return tee

# --- 6. Main Execution Block ---
if __name__ == "__main__":
    # --- Setup Experiment Directory ---
    cfg.OUTPUT_DIR = setup_experiment_directory()
    print(f"All outputs will be saved to: {cfg.OUTPUT_DIR}")
    
    # --- Setup Logging ---
    logger = setup_logging(cfg.OUTPUT_DIR)
    
    # --- Data Transforms ---
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),  # Added vertical flip
        transforms.RandomRotation(15),  # Increased from 10 to 15 degrees
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Added affine transforms
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),  # Increased intensity
        transforms.RandomGrayscale(p=0.1),  # Added grayscale conversion
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Added Gaussian blur
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.15))  # Added random erasing
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # --- Dataset and DataLoader Setup ---    
    try:
        train_annotations_df = pd.read_csv(cfg.TRAIN_ANNOTATIONS_FILE)
        val_annotations_df = pd.read_csv(cfg.VAL_ANNOTATIONS_FILE)
        test_annotations_df = pd.read_csv(cfg.TEST_ANNOTATIONS_FILE)
    except FileNotFoundError:
        print(f"Error: Annotation file not found at {cfg.TRAIN_ANNOTATIONS_FILE} or {cfg.VAL_ANNOTATIONS_FILE} or {cfg.TEST_ANNOTATIONS_FILE}")
        print("Please create an annotations.csv file and place your video frames in the data_root directory.")
        print("Refer to the comments in the code for the expected CSV format.")
        exit() # Exit if annotations file is not found

    train_dataset = EgocentricFrameDataset(
        annotations_df=train_annotations_df,
        data_root=cfg.DATA_ROOT,
        transform=train_transform,
        validate_images=cfg.VALIDATE_IMAGES
    )
    val_dataset = EgocentricFrameDataset(
        annotations_df=val_annotations_df,
        data_root=cfg.DATA_ROOT,
        transform=val_test_transform,
        validate_images=cfg.VALIDATE_IMAGES
    )
    test_dataset = EgocentricFrameDataset(
        annotations_df=test_annotations_df,
        data_root=cfg.DATA_ROOT,
        transform=val_test_transform,
        validate_images=cfg.VALIDATE_IMAGES
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, 
                             num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY, 
                             drop_last=True, collate_fn=safe_collate_fn,
                             prefetch_factor=cfg.PREFETCH_FACTOR if cfg.NUM_WORKERS > 0 else None,
                             persistent_workers=cfg.PERSISTENT_WORKERS if cfg.NUM_WORKERS > 0 else False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, 
                           num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY,
                           drop_last=False, collate_fn=safe_collate_fn,
                           prefetch_factor=cfg.PREFETCH_FACTOR if cfg.NUM_WORKERS > 0 else None,
                           persistent_workers=cfg.PERSISTENT_WORKERS if cfg.NUM_WORKERS > 0 else False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, 
                            num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY,
                            drop_last=False, collate_fn=safe_collate_fn,
                            prefetch_factor=cfg.PREFETCH_FACTOR if cfg.NUM_WORKERS > 0 else None,
                            persistent_workers=cfg.PERSISTENT_WORKERS if cfg.NUM_WORKERS > 0 else False)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")
    print(f"Effective batch size (with gradient accumulation): {cfg.BATCH_SIZE * cfg.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")
    print(f"Test batches per epoch: {len(test_loader)}")
    
    # Test if we can load a single batch
    print("Testing data loading...")
    try:
        print("  Creating iterator...")
        train_iterator = iter(train_loader)
        print("  Loading first batch...")
        test_batch = next(train_iterator)
        if test_batch is not None:
            images, labels = test_batch
            print(f"  Successfully loaded test batch: images shape {images.shape}, labels shape {labels.shape}")
            print(f"  Batch loaded successfully, proceeding with training...")
        else:
            print("  Warning: Test batch is None!")
        del train_iterator, test_batch  # Clean up
    except Exception as e:
        print(f"  Error loading test batch: {e}")
        print("  This might indicate a problem with the DataLoader configuration.")

    # Print GPU memory status
    if torch.cuda.is_available():
        print(f"GPU memory allocated before model creation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory reserved before model creation: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


    # --- Model, Loss, Optimizer ---
    model = MultiLabelClassifier(
        num_labels=cfg.NUM_LABELS, 
        backbone_name=cfg.BACKBONE_NAME, 
        pretrained=cfg.USE_PRETRAINED_BACKBONE
    )
    model.to(cfg.DEVICE)
    
    # Try to compile the model for better performance
    if cfg.USE_TORCH_COMPILE:
        try:
            model = torch.compile(model)
            print("Model compiled successfully for optimized execution")
        except Exception as e:
            print(f"Could not compile model (PyTorch 2.0+ required): {e}")
            print("Continuing with regular model...")

    # Print GPU memory status after model creation
    if torch.cuda.is_available():
        print(f"GPU memory allocated after model creation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory reserved after model creation: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    criterion = nn.BCEWithLogitsLoss()  # Changed from BCELoss to BCEWithLogitsLoss 
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=0.01)  # Changed to AdamW with weight decay
    
    # Calculate the number of training steps per epoch
    effective_steps_per_epoch = len(train_loader) // cfg.GRADIENT_ACCUMULATION_STEPS
    print(f"Training steps per epoch: {effective_steps_per_epoch}")
    
    # Use a more aggressive learning rate schedule to potentially increase GPU utilization
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=cfg.LEARNING_RATE * 5,  # Reduced from 10x to 5x to be less aggressive
        steps_per_epoch=effective_steps_per_epoch, 
        epochs=cfg.NUM_EPOCHS,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    print(f"OneCycleLR scheduler created: max_lr={cfg.LEARNING_RATE * 5}, steps_per_epoch={effective_steps_per_epoch}, epochs={cfg.NUM_EPOCHS}")

    # Initialize mixed precision scaler if available
    scaler = None
    if cfg.USE_MIXED_PRECISION and MIXED_PRECISION_AVAILABLE:
        try:
            scaler = GradScaler('cuda')
            print("Using mixed precision training with automatic scaling (new API)")
        except TypeError:
            # Fallback to older API
            scaler = GradScaler()
            print("Using mixed precision training with automatic scaling (legacy API)")
    else:
        print("Mixed precision training not available or disabled")

    # --- Early Stopping Setup ---
    early_stopping = EarlyStopping(
        patience=cfg.EARLY_STOPPING_PATIENCE,
        min_delta=cfg.EARLY_STOPPING_MIN_DELTA,
        mode=cfg.MONITOR_MODE,
        verbose=True
    )

    # --- Test Overfitting Capability (Debug) ---
    if cfg.DEBUG_MODE:
        print("\n" + "="*60)
        print("DEBUGGING: Testing overfitting capability...")
        print("="*60)
        
        # Create a copy of the model for testing
        test_model = MultiLabelClassifier(
            num_labels=len(cfg.LABELS), 
            backbone_name=cfg.BACKBONE_NAME, 
            pretrained=cfg.USE_PRETRAINED_BACKBONE
        )
        test_model.to(cfg.DEVICE)
        test_optimizer = optim.AdamW(test_model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=0.01)
        
        overfit_success = test_overfitting_capability(test_model, train_loader, criterion, test_optimizer, scaler)
        
        if not overfit_success:
            print("WARNING: Model cannot overfit on small sample. Check the training pipeline!")
            print("This suggests there might be issues with:")
            print("  - Learning rate (too low?)")
            print("  - Model architecture")
            print("  - Loss function")
            print("  - Gradient flow")
        
        # Clean up test model
        del test_model, test_optimizer
        torch.cuda.empty_cache()
        
        print("="*60)
        print("Continuing with full training...")
        print("="*60 + "\n")

    # --- Training Loop with Early Stopping ---
    best_val_score = float('-inf') if cfg.MONITOR_MODE == 'max' else float('inf')
    best_epoch = 0
    
    # Track training history for plotting
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_micro_f1': [],
        'val_macro_f1': [],
        'epochs': []
    }

    # Store initial model weights for debugging
    initial_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_weights[name] = param.data.clone()

    for epoch in range(1, cfg.NUM_EPOCHS + 1):            
        train_loss = train_model(model, train_loader, optimizer, criterion, epoch, scaler, scheduler)
        # Unpack the new return value: per_label_metrics
        val_loss, micro_f1, macro_f1, per_label_metrics = evaluate_model(model, val_loader, criterion)
        
        # Record training history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_micro_f1'].append(micro_f1)
        training_history['val_macro_f1'].append(macro_f1)
        training_history['epochs'].append(epoch)

        # Check if model weights are changing (debugging)
        if epoch <= 3:  # Only check for first few epochs to avoid spam
            weights_changed = False
            total_weight_change = 0
            for name, param in model.named_parameters():
                if param.requires_grad and name in initial_weights:
                    weight_diff = (param.data - initial_weights[name]).abs().sum().item()
                    total_weight_change += weight_diff
                    if weight_diff > 1e-6:
                        weights_changed = True
            
            print(f"Epoch {epoch}: Model weights changed: {weights_changed}, Total change: {total_weight_change:.6f}")
            
            if not weights_changed:
                print("WARNING: Model weights are not changing! There might be an issue with the optimizer or gradients.")

        # Print per-class metrics for validation
        print("  --- Per-Class Validation Metrics ---")
        for label, metrics in per_label_metrics.items():
            print(f"    {label}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, Accuracy={metrics['accuracy']:.4f}")
        print("  ----------------------------------")

        current_monitor_score = None
        if cfg.MONITOR_METRIC == 'validation_loss':
            current_monitor_score = val_loss
            # OneCycleLR doesn't use validation loss, so no scheduler.step() here
        elif cfg.MONITOR_METRIC == 'macro_f1':
            current_monitor_score = macro_f1
            # OneCycleLR doesn't use validation loss, so no scheduler.step() here
        elif cfg.MONITOR_METRIC == 'micro_f1':
            current_monitor_score = micro_f1
            # OneCycleLR doesn't use validation loss, so no scheduler.step() here
        else:
            raise ValueError("Unsupported MONITOR_METRIC in config.")

        early_stopping(current_monitor_score, model)

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}.")
            break
        
        if (cfg.MONITOR_MODE == 'max' and current_monitor_score > best_val_score) or \
           (cfg.MONITOR_MODE == 'min' and current_monitor_score < best_val_score):
            best_val_score = current_monitor_score
            best_epoch = epoch

    # --- Final Model Restoration and Test Set Evaluation ---
    if early_stopping.best_model_state:
        print(f"\nTraining complete. Restoring best model from epoch with {cfg.MONITOR_METRIC}: {early_stopping.best_score:.4f}...")
        model.load_state_dict(early_stopping.best_model_state)
        model_path = os.path.join(cfg.OUTPUT_DIR, "models", "final_best_multi_label_classifier.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Best model saved to: {model_path}")
    else:
        print("\nTraining complete. No improvement in validation metric observed from the start. Saving last model state.")
        model_path = os.path.join(cfg.OUTPUT_DIR, "models", "last_multi_label_classifier.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Last model saved to: {model_path}")

    # Create output directory (already created in setup, but just to be safe)
    if cfg.SAVE_PLOTS or cfg.SAVE_RESULTS:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Save training plots
    if cfg.SAVE_PLOTS:
        print("Saving training plots...")
        
        # Create training loss plot
        plt.figure(figsize=(12, 8))
        
        # Loss plot
        plt.subplot(2, 2, 1)
        plt.plot(training_history['epochs'], training_history['train_loss'], 'b-', label='Training Loss')
        plt.plot(training_history['epochs'], training_history['val_loss'], 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Micro F1 plot
        plt.subplot(2, 2, 2)
        plt.plot(training_history['epochs'], training_history['val_micro_f1'], 'g-', label='Validation Micro F1')
        plt.title('Validation Micro F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('Micro F1')
        plt.legend()
        plt.grid(True)
        
        # Macro F1 plot
        plt.subplot(2, 2, 3)
        plt.plot(training_history['epochs'], training_history['val_macro_f1'], 'm-', label='Validation Macro F1')
        plt.title('Validation Macro F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('Macro F1')
        plt.legend()
        plt.grid(True)
        
        # Combined metrics plot
        plt.subplot(2, 2, 4)
        plt.plot(training_history['epochs'], training_history['val_micro_f1'], 'g-', label='Micro F1')
        plt.plot(training_history['epochs'], training_history['val_macro_f1'], 'm-', label='Macro F1')
        plt.title('Validation F1 Scores Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(cfg.OUTPUT_DIR, "plots", f'training_plots_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training plots saved to: {plot_path}")

    print("\n--- Evaluating Model Performance on Test Set ---")
    # Unpack the new return value: test_per_label_metrics
    test_loss, test_micro_f1, test_macro_f1, test_per_label_metrics = evaluate_model(model, test_loader, criterion)

    print(f"\n--- Final Test Set Results ---")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Micro F1-score: {test_micro_f1:.4f}")
    print(f"Test Macro F1-score: {test_macro_f1:.4f}")

    print("\n--- Per-Class Test Metrics ---")
    for label, metrics in test_per_label_metrics.items():
        print(f"  {label}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, Accuracy={metrics['accuracy']:.4f}")
    print("----------------------------")

    # Save detailed results to JSON file
    if cfg.SAVE_RESULTS:
        print("Saving detailed results...")
        
        # Prepare comprehensive results dictionary
        results = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'batch_size': cfg.BATCH_SIZE,
                    'learning_rate': cfg.LEARNING_RATE,
                    'num_epochs': cfg.NUM_EPOCHS,
                    'image_size': cfg.IMAGE_SIZE,
                    'gradient_accumulation_steps': cfg.GRADIENT_ACCUMULATION_STEPS,
                    'effective_batch_size': cfg.BATCH_SIZE * cfg.GRADIENT_ACCUMULATION_STEPS,
                    'use_mixed_precision': cfg.USE_MIXED_PRECISION,
                    'monitor_metric': cfg.MONITOR_METRIC,
                    'early_stopping_patience': cfg.EARLY_STOPPING_PATIENCE
                },
                'dataset_info': {
                    'train_samples': len(train_dataset),
                    'val_samples': len(val_dataset),
                    'test_samples': len(test_dataset),
                    'labels': cfg.LABELS
                }
            },
            'training_history': training_history,
            'best_epoch': best_epoch,
            'best_validation_score': float(best_val_score),
            'early_stopping_triggered': early_stopping.early_stop,
            'final_test_results': {
                'test_loss': float(test_loss),
                'test_micro_f1': float(test_micro_f1),
                'test_macro_f1': float(test_macro_f1),
                'per_class_metrics': {
                    label: {
                        'precision': float(metrics['precision']),
                        'recall': float(metrics['recall']),
                        'f1': float(metrics['f1']),
                        'accuracy': float(metrics['accuracy'])
                    }
                    for label, metrics in test_per_label_metrics.items()
                }
            }
        }
        
        # Save results to JSON file
        results_path = os.path.join(cfg.OUTPUT_DIR, "results", f'training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to: {results_path}")
        
        # Also save a summary CSV for quick reference
        summary_data = []
        for label in cfg.LABELS:
            metrics = test_per_label_metrics[label]
            summary_data.append({
                'label': label,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'accuracy': metrics['accuracy']
            })
        
        # Add overall metrics
        summary_data.append({
            'label': 'OVERALL_MICRO',
            'precision': test_micro_f1,  # Note: this is F1, not precision
            'recall': test_micro_f1,     # Note: this is F1, not recall  
            'f1': test_micro_f1,
            'accuracy': test_micro_f1    # Note: this is F1, not accuracy
        })
        summary_data.append({
            'label': 'OVERALL_MACRO',
            'precision': test_macro_f1,  # Note: this is F1, not precision
            'recall': test_macro_f1,     # Note: this is F1, not recall
            'f1': test_macro_f1,
            'accuracy': test_macro_f1    # Note: this is F1, not accuracy
        })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(cfg.OUTPUT_DIR, "results", f'test_results_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"Test results summary saved to: {summary_path}")
        
        print(f"\nAll outputs saved to directory: {cfg.OUTPUT_DIR}")
    
    # --- Final Experiment Summary ---
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Experiment Directory: {cfg.OUTPUT_DIR}")
    print(f"Final Test Macro F1: {test_macro_f1:.4f}")
    print(f"Final Test Micro F1: {test_micro_f1:.4f}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Early Stopping: {'Yes' if early_stopping.early_stop else 'No'}")
    
    # List all generated files
    print("\nGenerated Files:")
    for root, dirs, files in os.walk(cfg.OUTPUT_DIR):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), cfg.OUTPUT_DIR)
            print(f"  {rel_path}")
    
    print("="*80)
    
    # Close the logger
    if 'logger' in locals():
        logger.close()
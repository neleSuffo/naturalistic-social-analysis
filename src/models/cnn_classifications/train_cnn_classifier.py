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
    print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# --- 1. Configuration and Hyperparameters ---
class Config:
    IMAGE_SIZE = 224
    BATCH_SIZE = 8  # Increased from 4 to 8 for higher GPU utilization
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Memory optimization settings
    NUM_WORKERS = 2  # Increased from 0 to use some CPU threads for data loading
    PIN_MEMORY = True  # Enable for faster GPU transfers
    GRADIENT_ACCUMULATION_STEPS = 4  # Reduced from 8 to 4 since we increased batch size
    
    # Enable mixed precision training
    USE_MIXED_PRECISION = True
    
    # GPU utilization settings
    USE_GRADIENT_CHECKPOINTING = True  # Save memory to allow larger batches
    PREFETCH_FACTOR = 2  # Number of batches to prefetch
    PERSISTENT_WORKERS = True  # Keep data loading workers alive between epochs
    
    # Memory clearing frequency
    MEMORY_CLEAR_FREQUENCY = 20  # Reduced frequency to allow more GPU utilization
    
    # Dataset validation
    VALIDATE_IMAGES = True  # Whether to check if all images exist during dataset creation
    
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

        # Shared feature extractor with slightly more capacity
        self.shared_features = nn.Sequential(
            nn.Linear(self.num_features, 512),  # Increased from 256 to 512 for more GPU utilization
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),  # Added an extra layer
            nn.ReLU(),
            nn.Dropout(0.3)
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
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training")
    for batch_idx, batch in enumerate(pbar):
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
                    loss += criterion(prediction_output, target_labels)
                
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
                loss += criterion(prediction_output, target_labels)
            
            # Scale loss for gradient accumulation
            loss = loss / cfg.GRADIENT_ACCUMULATION_STEPS
            loss.backward()
        
        accumulated_loss += loss.item()
        
        # Calculate accuracy metrics
        for i, label_name in enumerate(cfg.LABELS):
            target_labels = labels_tensor[:, i]
            prediction_output = outputs[label_name]
            # Apply sigmoid for evaluation since we removed it from the model
            preds = (torch.sigmoid(prediction_output) > 0.5).float()
            correct_predictions[label_name] += (preds == target_labels).sum().item()
            total_samples[label_name] += target_labels.size(0)

        # Perform gradient accumulation step
        if (batch_idx + 1) % cfg.GRADIENT_ACCUMULATION_STEPS == 0:
            if cfg.USE_MIXED_PRECISION and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
            total_loss += accumulated_loss
            accumulated_loss = 0

        pbar.set_postfix({'batch_loss': loss.item() * cfg.GRADIENT_ACCUMULATION_STEPS})
        
        # Aggressive memory management
        if batch_idx % cfg.MEMORY_CLEAR_FREQUENCY == 0:
            torch.cuda.empty_cache()
            gc.collect()
            
        # Delete variables to free memory immediately
        del images, labels_tensor, outputs, loss
        
    # Handle remaining gradients if batch size doesn't divide evenly
    if accumulated_loss > 0:
        if cfg.USE_MIXED_PRECISION and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        total_loss += accumulated_loss

    avg_loss = total_loss / (len(dataloader) / cfg.GRADIENT_ACCUMULATION_STEPS)
    
    accuracies = {label: (correct_predictions[label] / total_samples[label] if total_samples[label] > 0 else 0) * 100 for label in correct_predictions}
    
    print(f"Epoch {epoch} Average Training Loss: {avg_loss:.4f}")
    for label, acc in accuracies.items():
        print(f"  {label} Training Accuracy: {acc:.2f}%")
        
    if scheduler:
        scheduler.step(avg_loss)
        
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

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
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
                preds = (torch.sigmoid(prediction_output) > 0.5).float().cpu().numpy()
                targets = target_labels.cpu().numpy()
                
                all_preds[label_name].extend(preds)
                all_targets[label_name].extend(targets)

            total_loss += loss.item()
            
            # Less frequent memory clearing during evaluation to improve GPU utilization
            if batch_idx % (cfg.MEMORY_CLEAR_FREQUENCY * 2) == 0:
                torch.cuda.empty_cache()
                gc.collect()
                
            # Delete variables to free memory
            del images, labels_tensor, outputs, loss
            
        # Clear GPU cache after evaluation
        torch.cuda.empty_cache()
        gc.collect()

    avg_loss = total_loss / len(dataloader)

    # Calculate metrics for each label
    per_label_metrics = {}
    for label_name in cfg.LABELS:
        # Avoid division by zero if a class has no instances or no predictions
        if len(all_targets[label_name]) > 0:
            per_label_metrics[label_name] = {
                'accuracy': accuracy_score(all_targets[label_name], all_preds[label_name]),
                'precision': precision_score(all_targets[label_name], all_preds[label_name], zero_division=0),
                'recall': recall_score(all_targets[label_name], all_preds[label_name], zero_division=0),
                'f1': f1_score(all_targets[label_name], all_preds[label_name], zero_division=0)
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

# --- 6. Main Execution Block ---
if __name__ == "__main__":
    # --- Data Transforms ---
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
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

    # Print GPU memory status
    if torch.cuda.is_available():
        print(f"GPU memory allocated before model creation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory reserved before model creation: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


    # --- Model, Loss, Optimizer ---
    model = MultiLabelClassifier(num_labels=cfg.NUM_LABELS, backbone_name='resnet50', pretrained=True)
    model.to(cfg.DEVICE)

    # Print GPU memory status after model creation
    if torch.cuda.is_available():
        print(f"GPU memory allocated after model creation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU memory reserved after model creation: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    criterion = nn.BCEWithLogitsLoss()  # Changed from BCELoss to BCEWithLogitsLoss 
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

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

    # --- Training Loop with Early Stopping ---
    best_val_score = float('-inf') if cfg.MONITOR_MODE == 'max' else float('inf')
    best_epoch = 0

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        # Print memory status at start of each epoch
        if torch.cuda.is_available():
            print(f"Epoch {epoch} - GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
        train_loss = train_model(model, train_loader, optimizer, criterion, epoch, scaler, scheduler)
        # Unpack the new return value: per_label_metrics
        val_loss, micro_f1, macro_f1, per_label_metrics = evaluate_model(model, val_loader, criterion)

        # Print per-class metrics for validation
        print("  --- Per-Class Validation Metrics ---")
        for label, metrics in per_label_metrics.items():
            print(f"    {label}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, Accuracy={metrics['accuracy']:.4f}")
        print("  ----------------------------------")

        current_monitor_score = None
        if cfg.MONITOR_METRIC == 'validation_loss':
            current_monitor_score = val_loss
            scheduler.step(val_loss)
        elif cfg.MONITOR_METRIC == 'macro_f1':
            current_monitor_score = macro_f1
            scheduler.step(val_loss) 
        elif cfg.MONITOR_METRIC == 'micro_f1':
            current_monitor_score = micro_f1
            scheduler.step(val_loss) 
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
        torch.save(model.state_dict(), "final_best_multi_label_classifier.pth")
    else:
        print("\nTraining complete. No improvement in validation metric observed from the start. Saving last model state.")
        torch.save(model.state_dict(), "last_multi_label_classifier.pth")

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
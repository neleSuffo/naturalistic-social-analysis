import json
import os
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor, AdamW, get_scheduler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
from collections import defaultdict
import logging
from datetime import datetime
import csv
import time
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------
# CONFIG
# ----------------------------
DATA_DIR = "/home/nele_pauline_suffo/ProcessedData/quantex_videos_processed"
TRAIN_ANNOTATION_FILE = "/home/nele_pauline_suffo/ProcessedData/gaze_cls_input/gaze_cls_annotations_train.json"
VAL_ANNOTATION_FILE = "/home/nele_pauline_suffo/ProcessedData/gaze_cls_input/gaze_cls_annotations_val.json"
TEST_ANNOTATION_FILE = "/home/nele_pauline_suffo/ProcessedData/gaze_cls_input/gaze_cls_annotations_test.json"
BATCH_SIZE = 32  # Increased for more stable gradients
EPOCHS = 20  # More epochs for better convergence
LR = 1e-4  # Slightly higher learning rate
WEIGHT_DECAY = 0.01  # Add weight decay for regularization
WARMUP_STEPS = 100  # Add warmup steps

# Early stopping parameters
EARLY_STOPPING_PATIENCE = 10  # Increased patience
EARLY_STOPPING_MIN_DELTA = 0.001  # More sensitive improvement detection
EARLY_STOPPING_METRIC = "val_f1"  # Monitor F1 score for minority class focus

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = "vit_base_gaze_cls"
OUTPUT_DIR = f"/home/nele_pauline_suffo/outputs/gaze_classification/{model_name}_{timestamp}"

NUM_CLASSES = 2  # child_face vs adult_face (or gaze vs no-gaze)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
NUM_WORKERS = 6  # Reduced number of threads for data loading (can be adjusted)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(TRAIN_ANNOTATION_FILE), exist_ok=True)

# Setup output files
results_csv = os.path.join(OUTPUT_DIR, "results.csv")
train_log = os.path.join(OUTPUT_DIR, "train.log")
config_file = os.path.join(OUTPUT_DIR, "config.json")

# Configure file logging
file_handler = logging.FileHandler(train_log)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logging.getLogger().addHandler(file_handler)

# Save training configuration
config = {
    "model_name": model_name,
    "timestamp": timestamp,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LR,
    "num_classes": NUM_CLASSES,
    "device": str(DEVICE),
    "random_seed": RANDOM_SEED,
    "num_workers": NUM_WORKERS,
    "early_stopping": {
        "patience": EARLY_STOPPING_PATIENCE,
        "min_delta": EARLY_STOPPING_MIN_DELTA,
        "metric": EARLY_STOPPING_METRIC
    },
    "train_annotation_file": TRAIN_ANNOTATION_FILE,
    "val_annotation_file": VAL_ANNOTATION_FILE,
    "test_annotation_file": TEST_ANNOTATION_FILE,
    "model_architecture": "google/vit-base-patch16-224-in21k"
}

with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

# Initialize CSV results file
with open(results_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'epoch', 'time', 'train/loss', 'metrics/accuracy_top1', 'metrics/accuracy_top5', 
        'val/loss', 'val/accuracy_top1', 'lr/pg0', 'lr/pg1', 'lr/pg2'
    ])

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# Limit CPU threads for better resource management
torch.set_num_threads(NUM_WORKERS)
os.environ["OMP_NUM_THREADS"] = str(NUM_WORKERS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_WORKERS)
os.environ["MKL_NUM_THREADS"] = str(NUM_WORKERS)

logging.info(f"Using device: {DEVICE}")
logging.info(f"Number of CPU threads: {NUM_WORKERS}")
logging.info(f"PyTorch threads: {torch.get_num_threads()}")

# ----------------------------
# Dataset
# ----------------------------
class GazeDataset(Dataset):
    def __init__(self, annotation_file, feature_extractor, is_training=False):
        with open(annotation_file, 'r') as f:
            self.data = json.load(f)
        self.feature_extractor = feature_extractor
        self.is_training = is_training
        
        # Data augmentation for training
        if is_training:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),  # Increased rotation
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),  # More aggressive
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # More aggressive cropping
                transforms.RandomGrayscale(p=0.15),  # Increased chance
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),  # Add blur
            ])
        else:
            self.augment_transform = None
        
        # Filter out annotations with missing images
        valid_data = []
        for entry in self.data:
            if os.path.exists(entry["image_path"]):
                valid_data.append(entry)
            else:
                logging.warning(f"Image not found: {entry['image_path']}")
        
        self.data = valid_data
        logging.info(f"Loaded {len(self.data)} valid annotations from {annotation_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        img_path = entry["image_path"]
        
        try:
            image = Image.open(img_path).convert("RGB")
            bbox = entry["bbox"]  # [x1, y1, x2, y2]
            label = entry["label"]  # 0 or 1

            # Crop face region
            face = image.crop(bbox)
            
            # Ensure face crop is valid
            if face.size[0] == 0 or face.size[1] == 0:
                logging.warning(f"Invalid bbox {bbox} for image {img_path}")
                # Return a small black image as fallback
                face = Image.new('RGB', (224, 224), color='black')

            # Apply augmentation during training
            if self.is_training and self.augment_transform:
                face = self.augment_transform(face)

            # Preprocess using feature extractor
            features = self.feature_extractor(face, return_tensors="pt")

            return {
                "pixel_values": features["pixel_values"].squeeze(0),
                "label": torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            logging.error(f"Error processing image {img_path}: {e}")
            # Return a fallback black image
            face = Image.new('RGB', (224, 224), color='black')
            features = self.feature_extractor(face, return_tensors="pt")
            return {
                "pixel_values": features["pixel_values"].squeeze(0),
                "label": torch.tensor(0, dtype=torch.long)  # Default label
            }

# ----------------------------
# Load Feature Extractor and Model
# ----------------------------
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=NUM_CLASSES
).to(DEVICE)

# ----------------------------
# DataLoader
# ----------------------------
# Define datasets with data augmentation enabled for training
train_dataset = GazeDataset(TRAIN_ANNOTATION_FILE, feature_extractor, is_training=True)
val_dataset = GazeDataset(VAL_ANNOTATION_FILE, feature_extractor, is_training=False)

# Calculate class weights for imbalanced dataset
def calculate_class_weights(dataset):
    """Calculate class weights to handle imbalanced dataset"""
    labels = [item['label'].item() for item in dataset]
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    
    # Calculate weights inversely proportional to class frequency
    class_weights = total_samples / (len(class_counts) * class_counts)
    
    logging.info(f"Class distribution: {dict(enumerate(class_counts))}")
    logging.info(f"Class weights: {dict(enumerate(class_weights))}")
    
    return torch.FloatTensor(class_weights).to(DEVICE)

# Calculate class weights from training data
class_weights = calculate_class_weights(train_dataset)

# Focal Loss implementation for hard example mining
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Calculate cross entropy
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        
        # Calculate probabilities
        pt = torch.exp(-ce_loss)
        
        # Calculate focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Initialize focal loss with class weights
focal_loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

logging.info(f"Training dataset size: {len(train_dataset)}")
logging.info(f"Validation dataset size: {len(val_dataset)}")

# ----------------------------
# Optimizer and Scheduler
# ----------------------------
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=len(train_loader)*EPOCHS)

# ----------------------------
# Training Loop
# ----------------------------
def calculate_top5_accuracy(predictions, labels):
    """Calculate top-5 accuracy. For binary classification, top-5 is always 1.0"""
    return 1.0  # In binary classification, top-5 is always 100%

# Early stopping variables
best_metric = float('inf') if EARLY_STOPPING_METRIC == "val_loss" else 0.0
epochs_without_improvement = 0
best_epoch = 0
best_model_state = None

start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    
    model.train()
    losses = []
    all_preds = []
    all_labels = []
    all_logits = []
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        pixel_values = batch["pixel_values"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        
        # Use focal loss instead of standard cross entropy
        loss = focal_loss_fn(logits, labels)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_logits.append(logits.detach().cpu())

    # Calculate training metrics
    train_acc = accuracy_score(all_labels, all_preds)
    train_top5_acc = calculate_top5_accuracy(all_preds, all_labels)
    avg_loss = sum(losses) / len(losses)
    
    # Validation
    model.eval()
    val_losses = []
    val_preds = []
    val_labels = []
    val_logits = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation {epoch+1}/{EPOCHS}"):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            val_losses.append(loss.item())
            val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            val_logits.append(logits.detach().cpu())

    # Calculate validation metrics
    val_acc = accuracy_score(val_labels, val_preds)
    val_avg_loss = sum(val_losses) / len(val_losses)
    
    # Calculate F1 score for the minority class (gaze = class 1)
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(val_labels, val_preds, average=None)
    val_f1_macro = precision_recall_fscore_support(val_labels, val_preds, average='macro')[2]
    val_f1_gaze = val_f1[1] if len(val_f1) > 1 else 0.0  # F1 for gaze class
    
    # Early stopping logic
    if EARLY_STOPPING_METRIC == "val_loss":
        current_metric = val_avg_loss
    elif EARLY_STOPPING_METRIC == "val_acc":
        current_metric = val_acc
    elif EARLY_STOPPING_METRIC == "val_f1":
        current_metric = val_f1_gaze  # Focus on gaze class F1
    else:
        current_metric = val_f1_macro
    
    if EARLY_STOPPING_METRIC == "val_loss":
        improved = current_metric < (best_metric - EARLY_STOPPING_MIN_DELTA)
    else:  # val_acc or val_f1
        improved = current_metric > (best_metric + EARLY_STOPPING_MIN_DELTA)
    
    if improved:
        best_metric = current_metric
        best_epoch = epoch + 1
        epochs_without_improvement = 0
        # Save best model state
        best_model_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'train_loss': avg_loss,
            'train_acc': train_acc,
            'val_loss': val_avg_loss,
            'val_acc': val_acc,
            'config': config
        }
        logging.info(f"New best {EARLY_STOPPING_METRIC}: {best_metric:.5f} at epoch {best_epoch}")
    else:
        epochs_without_improvement += 1
        logging.info(f"No improvement for {epochs_without_improvement} epoch(s). Best {EARLY_STOPPING_METRIC}: {best_metric:.5f} at epoch {best_epoch}")
    
    # Get current learning rates from optimizer
    current_lr = optimizer.param_groups[0]['lr']
    
    # Calculate epoch time
    epoch_time = time.time() - epoch_start_time
    total_time = time.time() - start_time
    
    # Log to console
    print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_avg_loss:.4f}, Val Acc={val_acc:.4f}")
    
    # Log to file
    logging.info(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_loss:.5f}, Train Acc: {train_acc:.5f}, "
                f"Val Loss: {val_avg_loss:.5f}, Val Acc: {val_acc:.5f}, LR: {current_lr:.8f}, "
                f"Epoch Time: {epoch_time:.2f}s, Total Time: {total_time:.2f}s")
    
    # Write to CSV
    with open(results_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch + 1,
            round(total_time, 3),
            round(avg_loss, 5),
            round(train_acc, 5),
            round(train_top5_acc, 5),
            round(val_avg_loss, 5),
            round(val_acc, 5),
            f"{current_lr:.8f}",
            f"{current_lr:.8f}",
            f"{current_lr:.8f}"
        ])

    # Save checkpoint
    checkpoint_path = os.path.join(OUTPUT_DIR, f"model_epoch{epoch+1}.pt")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'train_loss': avg_loss,
        'train_acc': train_acc,
        'val_loss': val_avg_loss,
        'val_acc': val_acc,
        'config': config
    }, checkpoint_path)
    
    logging.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Check early stopping
    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        logging.info(f"Early stopping triggered after {epoch + 1} epochs")
        logging.info(f"Best {EARLY_STOPPING_METRIC}: {best_metric:.5f} at epoch {best_epoch}")
        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
        print(f"Best {EARLY_STOPPING_METRIC}: {best_metric:.5f} at epoch {best_epoch}")
        
        # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state['model_state_dict'])
            logging.info(f"Loaded best model from epoch {best_epoch}")
        
        # Save best model
        best_model_path = os.path.join(OUTPUT_DIR, "best_model.pt")
        torch.save(best_model_state, best_model_path)
        logging.info(f"Best model saved to: {best_model_path}")
        
        break

# Save final/best model if training completed without early stopping
if epochs_without_improvement < EARLY_STOPPING_PATIENCE:
    logging.info(f"Training completed all {EPOCHS} epochs")
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state_dict'])
        best_model_path = os.path.join(OUTPUT_DIR, "best_model.pt")
        torch.save(best_model_state, best_model_path)
        logging.info(f"Best model saved to: {best_model_path}")

# ----------------------------
# Test Set Evaluation
# ----------------------------
logging.info("Starting test set evaluation...")

test_dataset = GazeDataset(TEST_ANNOTATION_FILE, feature_extractor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

logging.info(f"Test dataset size: {len(test_dataset)}")

model.eval()
test_losses = []
test_preds = []
test_labels = []
test_logits = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        pixel_values = batch["pixel_values"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        test_losses.append(loss.item())
        test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        test_labels.extend(labels.cpu().numpy())
        test_logits.append(logits.detach().cpu())

# Calculate test metrics
test_acc = accuracy_score(test_labels, test_preds)
test_avg_loss = sum(test_losses) / len(test_losses)

# Calculate detailed metrics
precision, recall, f1, support = precision_recall_fscore_support(test_labels, test_preds, average=None)
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(test_labels, test_preds, average='macro')
precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(test_labels, test_preds, average='weighted')

# Confusion matrix
cm = confusion_matrix(test_labels, test_preds)

# Classification report
class_names = ['no_gaze', 'gaze']  # Adjust based on your classes
class_report = classification_report(test_labels, test_preds, target_names=class_names, output_dict=True)

# Log test results
logging.info(f"Test Evaluation Results:")
logging.info(f"Test Loss: {test_avg_loss:.5f}")
logging.info(f"Test Accuracy: {test_acc:.5f}")
logging.info(f"Macro-averaged Precision: {precision_macro:.5f}")
logging.info(f"Macro-averaged Recall: {recall_macro:.5f}")
logging.info(f"Macro-averaged F1: {f1_macro:.5f}")
logging.info(f"Weighted-averaged Precision: {precision_weighted:.5f}")
logging.info(f"Weighted-averaged Recall: {recall_weighted:.5f}")
logging.info(f"Weighted-averaged F1: {f1_weighted:.5f}")

# Print test results to console
print(f"\n{'='*50}")
print(f"TEST SET EVALUATION RESULTS")
print(f"{'='*50}")
print(f"Test Loss: {test_avg_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Macro-averaged Precision: {precision_macro:.4f}")
print(f"Macro-averaged Recall: {recall_macro:.4f}")
print(f"Macro-averaged F1: {f1_macro:.4f}")
print(f"Weighted-averaged Precision: {precision_weighted:.4f}")
print(f"Weighted-averaged Recall: {recall_weighted:.4f}")
print(f"Weighted-averaged F1: {f1_weighted:.4f}")

# Print per-class metrics
print(f"\nPer-class metrics:")
for i, class_name in enumerate(class_names):
    print(f"{class_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")

# Print confusion matrix
print(f"\nConfusion Matrix:")
print(f"{'':>12} {'Predicted':>20}")
print(f"{'Actual':>12} {'no_gaze':>10} {'gaze':>10}")
print(f"{'no_gaze':>12} {cm[0,0]:>10} {cm[0,1]:>10}")
print(f"{'gaze':>12} {cm[1,0]:>10} {cm[1,1]:>10}")

# Plot and save confusion matrix visualization
def plot_confusion_matrix(conf_matrix, class_names, output_dir):
    """
    Plot and save confusion matrix with counts and percentages.
    
    Parameters:
    - conf_matrix: numpy array of confusion matrix
    - class_names: list of class names
    - output_dir: directory to save the plot
    """
    # Note: sklearn confusion_matrix format is [True, Predicted]
    # We want Predicted on x-axis, True on y-axis, so we transpose
    conf_matrix_plot = conf_matrix.T
    
    # Compute percentages
    row_sums = conf_matrix_plot.sum(axis=1, keepdims=True)
    percentages = np.round((conf_matrix_plot / row_sums) * 100, 1)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(conf_matrix_plot, annot=False, fmt="d", cmap="Blues", cbar=True,
                     xticklabels=class_names, yticklabels=class_names)
    
    # Add text annotations with counts and percentages
    for i in range(conf_matrix_plot.shape[0]):
        for j in range(conf_matrix_plot.shape[1]):
            value = conf_matrix_plot[i, j]
            percent = percentages[i, j]
            
            # Set text color based on background intensity
            # Use white text for the darkest cell (highest value)
            max_value = np.max(conf_matrix_plot)
            text_color = "white" if value == max_value else "black"
            
            ax.text(j + 0.5, i + 0.5, f"{value}\n({percent}%)", 
                   ha="center", va="center", fontsize=12, color=text_color, weight='bold')
    
    plt.xlabel("Predicted", fontsize=12, weight='bold')
    plt.ylabel("True", fontsize=12, weight='bold')
    plt.title("Confusion Matrix for Gaze Classification", fontsize=14, weight='bold')
    
    # Save the figure
    plot_path = os.path.join(output_dir, "confusion_matrix_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    logging.info(f"Confusion matrix plot saved to {plot_path}")
    return plot_path

# Generate and save confusion matrix plot
cm_plot_path = plot_confusion_matrix(cm, class_names, OUTPUT_DIR)

# Save detailed test results
test_results = {
    "test_loss": round(test_avg_loss, 5),
    "test_accuracy": round(test_acc, 5),
    "macro_metrics": {
        "precision": round(precision_macro, 5),
        "recall": round(recall_macro, 5),
        "f1_score": round(f1_macro, 5)
    },
    "weighted_metrics": {
        "precision": round(precision_weighted, 5),
        "recall": round(recall_weighted, 5),
        "f1_score": round(f1_weighted, 5)
    },
    "per_class_metrics": {
        class_names[i]: {
            "precision": round(precision[i], 5),
            "recall": round(recall[i], 5),
            "f1_score": round(f1[i], 5),
            "support": int(support[i])
        } for i in range(len(class_names))
    },
    "confusion_matrix": {
        "matrix": cm.tolist(),
        "labels": class_names
    },
    "classification_report": class_report
}

# Save test results to JSON
test_results_file = os.path.join(OUTPUT_DIR, "test_results.json")
with open(test_results_file, 'w') as f:
    json.dump(test_results, f, indent=2)

# Save confusion matrix as CSV
cm_file = os.path.join(OUTPUT_DIR, "confusion_matrix.csv")
import pandas as pd
cm_df = pd.DataFrame(cm, index=[f"Actual_{name}" for name in class_names], 
                     columns=[f"Predicted_{name}" for name in class_names])
cm_df.to_csv(cm_file)

# Save detailed classification report as CSV
report_df = pd.DataFrame(class_report).transpose()
report_file = os.path.join(OUTPUT_DIR, "classification_report.csv")
report_df.to_csv(report_file)

logging.info(f"Test results saved to {test_results_file}")
logging.info(f"Confusion matrix saved to {cm_file}")
logging.info(f"Classification report saved to {report_file}")

# ----------------------------
# Save Final Model
# ----------------------------
final_time = time.time() - start_time
logging.info(f"Training completed in {final_time:.2f} seconds ({final_time/60:.2f} minutes)")

# Save final model and feature extractor
model.save_pretrained(OUTPUT_DIR)
feature_extractor.save_pretrained(OUTPUT_DIR)

# Create training summary
summary = {
    "training_completed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "total_training_time_seconds": round(final_time, 2),
    "total_training_time_minutes": round(final_time/60, 2),
    "total_epochs": EPOCHS,
    "final_train_loss": round(avg_loss, 5),
    "final_train_accuracy": round(train_acc, 5),
    "final_val_loss": round(val_avg_loss, 5),
    "final_val_accuracy": round(val_acc, 5),
    "test_performance": {
        "test_loss": round(test_avg_loss, 5),
        "test_accuracy": round(test_acc, 5),
        "test_precision_macro": round(precision_macro, 5),
        "test_recall_macro": round(recall_macro, 5),
        "test_f1_macro": round(f1_macro, 5),
        "test_precision_weighted": round(precision_weighted, 5),
        "test_recall_weighted": round(recall_weighted, 5),
        "test_f1_weighted": round(f1_weighted, 5)
    },
    "model_architecture": "google/vit-base-patch16-224-in21k",
    "num_parameters": sum(p.numel() for p in model.parameters()),
    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
}

# Save training summary
summary_file = os.path.join(OUTPUT_DIR, "training_summary.json")
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

logging.info(f"Training summary saved to {summary_file}")
logging.info(f"All outputs saved to: {OUTPUT_DIR}")
print(f"Training complete. Model and logs saved to: {OUTPUT_DIR}")
print(f"Training took {final_time:.2f} seconds ({final_time/60:.2f} minutes)")
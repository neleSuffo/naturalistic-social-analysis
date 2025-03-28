import torch
import logging
import torch.nn as nn
import torchvision.models as models
import torch.nn as nn
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import os
import datetime
import pandas as pd
from pathlib import Path
from constants import BasePaths, ResNetPaths
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Limit GPU memory usage
torch.cuda.set_per_process_memory_fraction(0.5, device=0)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

### Function Definitions

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train ResNet for classification")
    parser.add_argument("--target", type=str, required=True, choices=["gaze", "person", "face"],
                        help="Target classification task: 'gaze', 'person', or 'face'")
    return parser.parse_args()

def setup_model(target):
    """Set up the ResNet-152 model for binary classification."""
    logging.info(f"Loading pre-trained ResNet-152 model for {target} classification...")
    # Initialize the pre-trained ResNet-152 model
    #model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze specific layers (layer4 and fc)
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    
    # Modify the fully connected layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 1)
    )    
    # Move model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logging.info(f"Model moved to {device}")
    return model, device

def prepare_data(target, transform):
    """Prepare datasets and dataloaders for training and validation."""
    base_data_path = BasePaths.data_dir
    if target == "gaze":
        train_path = f"{base_data_path}/yolo_{target}_input/train"
        val_path = f"{base_data_path}/yolo_{target}_input/val"
    else:
        train_path = f"{base_data_path}/resnet_{target}_input/train"
        val_path = f"{base_data_path}/resnet_{target}_input/val"
    
    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    val_dataset = datasets.ImageFolder(val_path, transform=transform)
    
    # Map classes to binary labels (1 for positive class, 0 for others)
    positive_classes = {
        "gaze": "gaze",
        "person": "adult_person",
        "face": "adult_face" 
    }    
    positive_class = positive_classes[target]
        
    auto_class_to_idx = train_dataset.class_to_idx
    positive_idx = auto_class_to_idx.get(positive_class)
    if positive_idx is None:
        raise ValueError(f"Positive class '{positive_class}' not found in {auto_class_to_idx.keys()}")
   
    new_class_to_idx = {class_name: 1 if idx == positive_idx else 0 for class_name, idx in auto_class_to_idx.items()}
    train_dataset.class_to_idx = new_class_to_idx
    val_dataset.class_to_idx = new_class_to_idx
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    logging.info(f"Prepared data for {target}: Train size={len(train_dataset)}, Val size={len(val_dataset)}")
    logging.info(f"Class mapping: {new_class_to_idx}")

    return train_loader, val_loader

def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """Train the model for one epoch and return the average loss."""
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = (torch.sigmoid(outputs).cpu().numpy() > 0.5).astype(int)
            y_true.extend(labels.cpu().numpy().flatten())  # Flatten to scalars
            y_pred.extend(preds.flatten())                 # Flatten to scalars
    return val_loss / len(val_loader), y_true, y_pred

def calculate_metrics(y_true, y_pred):
    """Calculate and return evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1)
    rec = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    return acc, prec, rec, f1

def create_output_dir(base_output_dir: Path, target: str) -> Path:
    """Create timestamped output directory for storing metrics."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created output directory: {output_dir}")
    return output_dir

def plot_results(train_losses, val_losses, accuracies, precisions, recalls, f1_scores, output_dir, target):
    """Plot and save training/validation loss and metric curves."""
    # Loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss Curves for {target} Classification")
    plt.grid()
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()

    # Metrics curve
    plt.figure(figsize=(12, 6))
    epochs = np.arange(1, len(accuracies) + 1)
    plt.plot(epochs, accuracies, label="Accuracy", marker="o")
    plt.plot(epochs, precisions, label="Precision", marker="o")
    plt.plot(epochs, recalls, label="Recall", marker="o")
    plt.plot(epochs, f1_scores, label="F1 Score", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.title(f"Metrics for {target} Classification")
    plt.grid()
    plt.savefig(output_dir / "metrics_curve.png")
    plt.close()

    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'epoch': range(1, len(accuracies) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'accuracy': accuracies,
        'precision': precisions,
        'recall': recalls,
        'f1_score': f1_scores
    })
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    
def main():
    """Orchestrate the training process."""
    # Parse arguments and set up paths
    args = parse_args()
    target = args.target
    base_output_dir = getattr(ResNetPaths, f"{args.target}_output_dir")
    model_save_path = getattr(ResNetPaths, f"{args.target}_trained_weights_path")

    # Create timestamped output directory
    output_dir = create_output_dir(Path(base_output_dir), target)
    
    # Save training configuration
    with open(output_dir / "config.txt", "w") as f:
        f.write(f"Target: {target}\n")
        f.write(f"Training started: {datetime.datetime.now()}\n")
        f.write(f"Model save path: {model_save_path}\n")
        
    # Initialize model and data
    model, device = setup_model(target)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_loader, val_loader = prepare_data(target, transform)

    # Set up training components
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    scaler = torch.amp.GradScaler()

    # Training loop
    num_epochs = 100
    patience = 7
    best_val_loss = float("inf")
    early_stop_counter = 0
    train_losses, val_losses = [], []
    accuracies, precisions, recalls, f1_scores = [], [], [], []

    for epoch in range(num_epochs):
        start_time = time.time()

        avg_train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        train_losses.append(avg_train_loss)

        avg_val_loss, y_true, y_pred = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(avg_val_loss)

        acc, prec, rec, f1 = calculate_metrics(y_true, y_pred)
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

        elapsed_time = time.time() - start_time
        logging.info(f"Epoch [{epoch+1}/{num_epochs}] - {target} Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} (Time: {elapsed_time:.2f}s)")
        logging.info(f"Metrics - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

        scheduler.step()
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Best model saved with Val Loss: {best_val_loss:.4f}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                logging.info(f"Early stopping triggered for {target}.")
                break

    # Save final metrics
    final_metrics = {
        'best_val_loss': best_val_loss,
        'final_accuracy': accuracies[-1],
        'final_precision': precisions[-1],
        'final_recall': recalls[-1],
        'final_f1': f1_scores[-1],
        'epochs_trained': epoch + 1
    }
    
    with open(output_dir / "final_metrics.txt", "w") as f:
        for metric, value in final_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
            
    logging.info(f"Training complete for {target}. Best model at: {model_save_path}")
    plot_results(train_losses, val_losses, accuracies, precisions, recalls, f1_scores, output_dir, target)

if __name__ == "__main__":
    main()
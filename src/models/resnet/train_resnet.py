import torch
import logging
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import os
from constants import BasePaths

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
    model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  # Binary classification
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
    positive_classes = {"gaze": "gaze", "person": "adult", "face": "adult"}
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
    return train_loader, val_loader

def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """Train the model for one epoch and return the average loss."""
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion, device):
    """Validate the model for one epoch and return loss and predictions."""
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
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds)
    return val_loss / len(val_loader), y_true, y_pred

def calculate_metrics(y_true, y_pred):
    """Calculate and return evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1)
    rec = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    return acc, prec, rec, f1

def plot_results(train_losses, val_losses, accuracies, precisions, recalls, f1_scores, output_dir, target):
    """Plot and save training/validation loss and metric curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss Curves for {target} Classification")
    plt.grid()
    plt.savefig(f"{output_dir}/loss_curve.png")
    plt.close()

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
    plt.savefig(f"{output_dir}/metrics_curve.png")
    plt.close()

def main():
    """Orchestrate the training process."""
    # Parse arguments and set up paths
    args = parse_args()
    target = args.target
    output_dir = f"{BasePaths.output_dir}/resnet_{target}_classification"
    model_save_path = f"{BasePaths.models_dir}/resnet_{target}.pth"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model and data
    model, device = setup_model(target)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_loader, val_loader = prepare_data(target, transform)

    # Set up training components
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = torch.amp.GradScaler()

    # Training loop
    num_epochs = 100
    patience = 10
    best_val_loss = float("inf")
    early_stop_counter = 0
    train_losses, val_losses = [], []
    accuracies, precisions, recalls, f1_scores = [], [], []

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

        scheduler.step(avg_val_loss)
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

    logging.info(f"Training complete for {target}. Best model at: {model_save_path}")
    plot_results(train_losses, val_losses, accuracies, precisions, recalls, f1_scores, output_dir, target)

if __name__ == "__main__":
    main()
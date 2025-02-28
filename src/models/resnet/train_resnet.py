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
from constants import ResNetPaths

# Limit GPU memory usage
torch.cuda.set_per_process_memory_fraction(0.5, device=0)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load Pretrained ResNet-50
logging.info("Loading pre-trained ResNet-50 model...")
resnet152 = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)

# Modify the last FC layer for binary classification
num_ftrs = resnet152.fc.in_features
resnet152.fc = nn.Linear(num_ftrs, 1)  # Single neuron for binary classification
logging.info(f"Modified last layer: {resnet152.fc}")

# Use Sigmoid activation
model = resnet152

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Combines sigmoid + BCELoss safely
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# Enable Automatic Mixed Precision (AMP)
scaler = torch.amp.GradScaler()

# Define transforms for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_path = "/home/nele_pauline_suffo/ProcessedData/yolo_gaze_input/train"
val_path = "/home/nele_pauline_suffo/ProcessedData/yolo_gaze_input/val"
train_dataset = datasets.ImageFolder(train_path, transform=transform)
val_dataset = datasets.ImageFolder(val_path, transform=transform)

# Set class-to-index mapping
train_dataset.class_to_idx = {"gaze": 1, "no_gaze": 0}
val_dataset.class_to_idx = {"gaze": 1, "no_gaze": 0}

logging.info(f"Training dataset size: {len(train_dataset)} images")
logging.info(f"Validation dataset size: {len(val_dataset)} images")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Training setup
num_epochs = 100
patience = 10
best_val_loss = float("inf")
early_stop_counter = 0
model_save_path = ResNetPaths.trained_weights_path

# Store metrics for visualization
train_losses, val_losses = [], []
accuracies, precisions, recalls, f1_scores = [], [], [], []

logging.info(f"Starting training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()  # Track epoch time

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)  # Reshape labels

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():  # Enable mixed precision
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # Print loss every 10 steps
        if (batch_idx + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_batch = elapsed_time / (batch_idx + 1)
            remaining_batches = len(train_loader) - (batch_idx + 1)
            eta = avg_time_per_batch * remaining_batches
            logging.info(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], "
                f"Loss: {loss.item():.4f}, ETA: {eta:.2f}s"
            )

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation Phase
    model.eval()
    val_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = (torch.sigmoid(outputs).cpu().numpy() > 0.5).astype(int)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds)

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1)
    rec = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)

    elapsed_time = time.time() - start_time
    logging.info(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} (Time: {elapsed_time:.2f}s)")
    logging.info(f"Validation Metrics - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")

    # Adjust learning rate
    scheduler.step(avg_val_loss)

    # Early Stopping & Checkpointing
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), model_save_path)
        logging.info(f"New best model saved with Val Loss: {best_val_loss:.4f}")
    else:
        early_stop_counter += 1
        logging.info(f"No improvement. Early stop counter: {early_stop_counter}/{patience}")

    if early_stop_counter >= patience:
        logging.info("Early stopping triggered. Training stopped.")
        break

logging.info(f"Training complete! Best model saved at: {model_save_path}")

# Plot Loss Curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss", marker="o")
plt.plot(val_losses, label="Validation Loss", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Validation Loss")
plt.grid()
plt.savefig("/home/nele_pauline_suffo/outputs/resnet_gaze_classification/loss_curve.png")
plt.show()

# Plot Accuracy, Precision, Recall, and F1-Score
plt.figure(figsize=(12, 6))
epochs = np.arange(1, len(accuracies) + 1)

plt.plot(epochs, accuracies, label="Accuracy", marker="o")
plt.plot(epochs, precisions, label="Precision", marker="o")
plt.plot(epochs, recalls, label="Recall", marker="o")
plt.plot(epochs, f1_scores, label="F1 Score", marker="o")

plt.xlabel("Epochs")
plt.ylabel("Metric Value")
plt.legend()
plt.title("Model Performance Metrics")
plt.grid()
plt.savefig("metrics_curve.png")
plt.show()
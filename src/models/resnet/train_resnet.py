import torch
import logging
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from constants import ResNetPaths

torch.cuda.set_per_process_memory_fraction(0.5, device=0)  # Use only 50% of GPU 0 memory

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load Pretrained ResNet-50
logging.info("Loading pre-trained ResNet-50 model...")
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Modify the last FC layer for binary classification
num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_ftrs, 1)  # Single neuron for binary classification
logging.info(f"Modified last layer: {resnet50.fc}")

# Use Sigmoid activation
model = resnet50

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # This combines sigmoid + BCELoss safely
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# Enable Automatic Mixed Precision (AMP)
scaler = torch.amp.GradScaler(device_type="cuda")

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

logging.info(f"Starting training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)  # Reshape labels

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():  # Enable mixed precision
            outputs = model(images)
            loss = criterion(outputs, labels)
sc
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            logging.info(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_train_loss = running_loss / len(train_loader)

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

            preds = (outputs.cpu().numpy() > 0.5).astype(int)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds)

    avg_val_loss = val_loss / len(val_loader)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    logging.info(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
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
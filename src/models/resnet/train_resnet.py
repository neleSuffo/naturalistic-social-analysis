import torch
import logging
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
model = nn.Sequential(resnet50, nn.Sigmoid())

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Define transforms for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet-50 expects 224x224 images
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
num_epochs = 100  # Max epochs
patience = 10  # Early stopping patience
best_val_loss = float("inf")
early_stop_counter = 0
model_save_path = "/home/nele_pauline_suffo/models/resnet_gaze_classification.pth"

logging.info(f"Starting training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)  # Reshape labels
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:  # Log every 10 batches
            logging.info(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_train_loss = running_loss / len(train_loader)
    
    # Validation Phase
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    logging.info(f"Epoch [{epoch+1}/{num_epochs}] completed. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Early Stopping Logic
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), model_save_path)
        logging.info(f"New best model saved with Val Loss: {best_val_loss:.4f}")
    else:
        early_stop_counter += 1
        logging.info(f"No improvement in validation loss. Early stop counter: {early_stop_counter}/{patience}")

    if early_stop_counter >= patience:
        logging.info("Early stopping triggered. Training stopped.")
        break

logging.info(f"Training complete! Best model saved at: {model_save_path}")
            

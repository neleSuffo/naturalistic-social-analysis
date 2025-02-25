import torch
import logging
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

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

# Training loop
num_epochs = 10
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

    avg_loss = running_loss / len(train_loader)
    logging.info(f"Epoch [{epoch+1}/{num_epochs}] completed. Avg Loss: {avg_loss:.4f}")

logging.info("Training complete!")

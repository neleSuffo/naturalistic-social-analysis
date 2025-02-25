import torch
import logging
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
from PIL import Image

# Load Pretrained ResNet-50
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)  

# Modify the last FC layer for binary classification
num_ftrs = resnet50.fc.in_features  # Get input features of FC layer
resnet50.fc = nn.Linear(num_ftrs, 1)  # Output 1 neuron for binary classification

# Use Sigmoid activation
model = nn.Sequential(resnet50, nn.Sigmoid())

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Define transforms for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet-50 expects 224x224 images
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load your dataset (Replace with your own dataset path)
train_dataset = datasets.ImageFolder("/home/nele_pauline_suffo/ProcessedData/yolo_gaze_input/train", transform=transform)
val_dataset = datasets.ImageFolder("/home/nele_pauline_suffo/ProcessedData/yolo_gaze_input/val", transform=transform)

# Set class_to_idx mapping
train_dataset.class_to_idx = {"gaze": 1, "no_gaze": 0}  

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)  # Reshape labels
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    logging(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

logging("Training complete!")


import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from constants import ResNetPaths
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model architecture
logging.info("Loading model architecture...")
resnet152 = models.resnet152(weights=None)
num_ftrs = resnet152.fc.in_features
resnet152.fc = nn.Linear(num_ftrs, 1)

# Load trained weights
model_path = ResNetPaths.trained_weights_path
if os.path.exists(model_path):
    state_dict = torch.load(model_path, map_location=device)
    model = resnet152
    
    try:
        model.load_state_dict(state_dict)
        logging.info(f"Model loaded from {model_path}")
    except RuntimeError as e:
        logging.error(f"Failed to load model: {str(e)}")
        exit()
else:
    logging.error(f"Model file {model_path} not found!")
    exit()

# Add Sigmoid to model for evaluation
model = nn.Sequential(model, nn.Sigmoid())  # Ensures sigmoid is applied to the output
model = model.to(device)
model.eval()  # Set model to evaluation mode

# Define transformations for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the validation dataset
val_path = "/home/nele_pauline_suffo/ProcessedData/yolo_gaze_input/test"  # Adjust path if necessary
val_dataset = datasets.ImageFolder(val_path, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

logging.info(f"Validation dataset size: {len(val_dataset)} images")

# Evaluate the model
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        predictions = (outputs.squeeze() >= 0.5).float()  # Add squeeze() here
        
        # Move to CPU and convert to numpy arrays immediately
        y_true.extend(labels.cpu().numpy().flatten())  # Add flatten()
        y_pred.extend(predictions.cpu().numpy().flatten())  # Add flatten()

# Convert lists to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Compute metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

logging.info(f"Validation Accuracy: {accuracy:.4f}")
logging.info(f"Precision (Gaze): {precision:.4f}, Recall (Gaze): {recall:.4f}, F1 Score (Gaze): {f1:.4f}")

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot and save the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Gaze", "No Gaze"], yticklabels=["Gaze", "No Gaze"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

# Create output directory if it does not exist
conf_matrix_path = Path(ResNetPaths.confusion_matrix_path)
conf_matrix_path.parent.mkdir(parents=True, exist_ok=True)

# Save the confusion matrix to a file
plt.savefig(conf_matrix_path)
logging.info(f"Confusion matrix saved to {conf_matrix_path}")

# Show the plot (optional, can be removed for headless execution)
plt.show()

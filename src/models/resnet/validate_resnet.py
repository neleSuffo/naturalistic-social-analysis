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
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Evaluate ResNet model for a specific target.")
parser.add_argument("--target", choices=["person", "face", "gaze"], required=True, help="Target category: person, face, or gaze.")
args = parser.parse_args()

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
model_path = getattr(ResNetPaths, f"{args.target}_trained_weights_path")
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
model = nn.Sequential(model, nn.Sigmoid())
model = model.to(device)
model.eval()

# Define transformations for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the validation dataset
val_path = f"/home/nele_pauline_suffo/ProcessedData/resnet_{args.target}_input/test"
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
        predictions = (outputs.squeeze() >= 0.5).float()
        
        y_true.extend(labels.cpu().numpy().flatten())
        y_pred.extend(predictions.cpu().numpy().flatten())

# Convert lists to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Compute metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

logging.info(f"Validation Accuracy: {accuracy:.4f}")
logging.info(f"Precision ({args.target}): {precision:.4f}, Recall ({args.target}): {recall:.4f}, F1 Score ({args.target}): {f1:.4f}")

# Define class labels for each target type
class_labels = {
    'person': ['child_person', 'adult_person'],
    'face': ['child_face', 'adult_face'],
    'gaze': ['no_gaze', 'gaze']
}

# Get current target's labels
target_labels = class_labels[args.target]

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot and save the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=target_labels,
    yticklabels=target_labels
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix - {args.target.capitalize()}")

# Adjust layout for better label visibility
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Create output directory if it does not exist
output_dir = getattr(ResNetPaths, f"{args.target}_output_dir")
conf_matrix_path = output_dir / "confusion_matrix.png"
conf_matrix_path.parent.mkdir(parents=True, exist_ok=True)

# Save the confusion matrix to a file
plt.savefig(conf_matrix_path)
logging.info(f"Confusion matrix saved to {conf_matrix_path}")

# Show the plot (optional)
plt.show()

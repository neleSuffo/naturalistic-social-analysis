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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
from constants import ResNetPaths
from pathlib import Path
import argparse
from datetime import datetime

# parse Command-Line Arguments
parser = argparse.ArgumentParser(description="Evaluate ResNet model for a specific target.")
parser.add_argument("--target", choices=["person", "face", "gaze"], required=True, help="Target category: person, face, or gaze.")
args = parser.parse_args()

# configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define Device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Pretrained ResNet Model
logging.info("Loading ResNet model...")
resnet50 = models.resnet50(weights=None)
num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 1)
)

# Load Trained Weights
model_path = getattr(ResNetPaths, f"{args.target}_trained_weights_path")
if os.path.exists(model_path):
    try:
        state_dict = torch.load(model_path, map_location=device)
        # Remove unexpected keys from state dict
        for key in list(state_dict.keys()):
            if key.startswith('fc.'):
                state_dict.pop(key)
        
        model = resnet50
        model.load_state_dict(state_dict, strict=False)
        logging.info(f"Model loaded from {model_path}")
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        exit(1)
else:
    logging.error(f"Model file {model_path} not found!")
    exit(1)

# Prepare Model for Evaluation
model = nn.Sequential(model, nn.Sigmoid())  # Add Sigmoid activation for binary classification
model = model.to(device)
model.eval()

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# load Validation Dataset
if args.target == "gaze":
    val_path = f"/home/nele_pauline_suffo/ProcessedData/yolo_{args.target}_input/test"
else:
    val_path = f"/home/nele_pauline_suffo/ProcessedData/resnet_{args.target}_input/test"
val_dataset = datasets.ImageFolder(val_path, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

logging.info(f"Validation dataset size: {len(val_dataset)} images")

# Run Model Evaluation
y_true, y_pred = [], []

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

# Compute Performance Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

logging.info(f"Validation Accuracy: {accuracy:.4f}")
logging.info(f"Precision ({args.target}): {precision:.4f}, Recall ({args.target}): {recall:.4f}, F1 Score ({args.target}): {f1:.4f}")

# Define Class Labels
class_labels = {
    'person': ['adult_person' , 'child_person'],
    'face': ['adult_face', 'child_face', ],
    'gaze': ['gaze', 'no_gaze']
}
target_labels = class_labels[args.target]

# Create Output Directory
base_output_dir = getattr(ResNetPaths, f"{args.target}_output_dir")
current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(base_output_dir) / f"{current_date}_validation"
output_dir.mkdir(parents=True, exist_ok=True)

# Save Metrics to a File
metrics_path = output_dir / "metrics.txt"
with open(metrics_path, "w") as f:
    f.write(f"Validation Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision ({args.target}): {precision:.4f}\n")
    f.write(f"Recall ({args.target}): {recall:.4f}\n")
    f.write(f"F1 Score ({args.target}): {f1:.4f}\n")

logging.info(f"Metrics saved to {metrics_path}")

# Generate and Save Confusion Matrix Plot
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_labels, yticklabels=target_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix - {args.target.capitalize()}")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

conf_matrix_path = output_dir / "confusion_matrix.png"
plt.savefig(conf_matrix_path)
logging.info(f"Confusion matrix saved to {conf_matrix_path}")

# Save Prediction Histogram
plt.figure(figsize=(8, 6))
plt.hist(y_pred, bins=20, color="blue", alpha=0.7, edgecolor="black")
plt.xlabel("Prediction Confidence (0 = Negative, 1 = Positive)")
plt.ylabel("Count")
plt.title(f"Histogram of Predictions - {args.target.capitalize()}")

hist_path = output_dir / "predictions_histogram.png"
plt.savefig(hist_path)
logging.info(f"Prediction histogram saved to {hist_path}")

# Save Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred)
plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, marker="o", color="red")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curve - {args.target.capitalize()}")
plt.grid()

pr_curve_path = output_dir / "precision_recall_curve.png"
plt.savefig(pr_curve_path)
logging.info(f"Precision-Recall curve saved to {pr_curve_path}")
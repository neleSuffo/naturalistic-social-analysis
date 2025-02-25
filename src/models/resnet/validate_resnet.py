import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from constants import ResNetPaths

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model architecture
logging.info("Loading model architecture...")
resnet50 = models.resnet50(weights=None)
num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_ftrs, 1)

# Load trained weights
model_path = ResNetPaths.trained_weights_path
if os.path.exists(model_path):
    state_dict = torch.load(model_path, map_location=device)
    model = resnet50
    
    try:
        model.load_state_dict(state_dict)
        logging.info(f"Model loaded from {model_path}")
    except RuntimeError as e:
        logging.error(f"Failed to load model: {str(e)}")
        exit()
else:
    logging.error(f"Model file {model_path} not found!")
    exit()

# Add Sigmoid to model for evaluation if necessary
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

with torch.no_grad():  # No gradient computation needed for evaluation
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)  # Keep labels on the same device as the model
        
        # Perform forward pass
        outputs = model(images)  # Outputs are logits transformed by sigmoid already
        predictions = (outputs >= 0.5).float()  # Convert to binary predictions

        # Collect true labels and predictions
        y_true.extend(labels.cpu().numpy())  # Labels should be moved to CPU for metrics calculation
        y_pred.extend(predictions.cpu().numpy())  # Predictions should be moved to CPU for metrics calculation

# Compute metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

logging.info(f"Validation Accuracy: {accuracy:.4f}")
logging.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

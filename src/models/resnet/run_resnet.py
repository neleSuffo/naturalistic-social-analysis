import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import logging
import os
import argparse
from constants import ResNetPaths

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define device (CUDA if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Perform inference on an image using a ResNet model')
parser.add_argument('--target', type=str, required=True, choices=['gaze', 'person', 'face'],
                    help='Target model to use for inference (gaze, person, face)')
parser.add_argument('--image_path', type=str, required=True,
                    help='Path to the image to be inferred')
args = parser.parse_args()

# Load model architecture
logging.info("Loading model architecture...")
resnet152 = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
num_ftrs = resnet152.fc.in_features  # Get the number of input features to the final FC layer

# Modify the last fully connected (fc) layer for binary classification
resnet152.fc = nn.Linear(num_ftrs, 1)  # Change output layer to 1 (binary classification)

# Load trained weights dynamically based on the target
model_path = getattr(ResNetPaths, f"{args.target}_trained_weights_path")
if os.path.exists(model_path):
    state_dict = torch.load(model_path, map_location=device)
    state_dict.pop('fc.weight', None)
    state_dict.pop('fc.bias', None)
    resnet152.load_state_dict(state_dict, strict=False)
    logging.info(f"Model loaded from {model_path}")
else:
    logging.error(f"Model file {model_path} not found!")
    exit()

# Add Sigmoid activation to the model for inference
model = nn.Sequential(resnet152, nn.Sigmoid())
model = model.to(device)
model.eval()

# Define transformations for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the image
image = Image.open(args.image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)  # Apply transformations and move to device

# Perform inference
with torch.no_grad():
    output = model(image)
    prediction = (output >= 0.5).float()

# Convert the prediction to class label
prediction_label = args.target if prediction.item() == 1 else f"no_{args.target}"
logging.info(f"Predicted label: {prediction_label}")

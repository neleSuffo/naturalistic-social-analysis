import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import logging
import os
from constants import ResNetPaths

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define device (CUDA if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model architecture
logging.info("Loading model architecture...")
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)  # Load with pre-trained weights
num_ftrs = resnet50.fc.in_features  # Get the number of input features to the final FC layer

# Modify the last fully connected (fc) layer for binary classification
resnet50.fc = nn.Linear(num_ftrs, 1)  # Change output layer to 1 (binary classification)

# Load trained weights
model_path = ResNetPaths.trained_weights_path  # Define the path for your trained weights
if os.path.exists(model_path):
    # Load the state dict, ignoring the `fc` layer mismatch
    state_dict = torch.load(model_path, map_location=device)  # Load the saved state dict
    
    # Remove the `fc` layer weights from the state_dict (for binary classification)
    state_dict.pop('fc.weight', None)  # Remove weights for fc
    state_dict.pop('fc.bias', None)    # Remove bias for fc

    # Load the state dict into the model, ignoring `fc` layer weights
    resnet50.load_state_dict(state_dict, strict=False)
    
    logging.info(f"Model loaded from {model_path}")
else:
    logging.error(f"Model file {model_path} not found!")  # Exit if model weights are not found
    exit()

# Add Sigmoid activation to the model for inference (applied to final output)
model = nn.Sequential(resnet50, nn.Sigmoid())  # Apply Sigmoid after ResNet for binary classification
model = model.to(device)  # Move model to GPU or CPU based on availability
model.eval()  # Set model to evaluation mode

# Define transformations for input images (same preprocessing as used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224 pixels
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
])

# Load the image to be inferred
image_path = "/home/nele_pauline_suffo/ProcessedData/yolo_gaze_input/test/gaze/quantex_at_home_id261610_2022_04_01_01_036210_face_0.jpg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")  # Open and convert image to RGB

# Apply transformations to the image (resize, convert to tensor, normalize)
image = transform(image).unsqueeze(0)  # Add batch dimension (model expects a batch)

# Move the image to the same device as the model (GPU or CPU)
image = image.to(device)

# Perform inference
with torch.no_grad():  # No gradient computation needed during inference
    output = model(image)  # Forward pass through the model
    prediction = (output >= 0.5).float()  # Convert logits to binary prediction (0 or 1)

# Convert the prediction to class label
prediction_label = "gaze" if prediction.item() == 1 else "no_gaze"

# Log the result
logging.info(f"Predicted label: {prediction_label}")  # Log the prediction

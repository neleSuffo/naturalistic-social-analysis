import os
import torch
import logging
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# Load the model
def load_model(model_path, num_classes):
    """
    Load the saved gaze classification model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model with correct architecture
    model = models.efficientnet_b3(pretrained=False)
    
    # Recreate the same classifier structure used in training
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# Preprocess the input image
def preprocess_image(image_path):
    """
    Preprocess the input image for inference.

    Args:
        image_path (str): Path to the input image.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

# Perform inference
def classify_gaze(model, image_tensor, class_names):
    """
    Perform gaze classification on an input image tensor.

    Args:
        model (torch.nn.Module): Trained model.
        image_tensor (torch.Tensor): Preprocessed image tensor.
        class_names (list): List of class names.

    Returns:
        str: Predicted class.
        dict: Class probabilities.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()

    predicted_class = class_names[probabilities.argmax()]
    class_probabilities = {class_name: prob for class_name, prob in zip(class_names, probabilities)}

    return predicted_class, class_probabilities

def main(image_path):
    # Paths and configuration
    model_path = "/home/nele_pauline_suffo/outputs/efficientnet/20250120_193131/best_model.pth"
    class_names = ["No Gaze", "Gaze"]  # Update as per your dataset
    num_classes = len(class_names)

    # Verify path exists before processing
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")
    
    # Load model
    model = load_model(model_path, num_classes)

    # Preprocess image
    image_tensor = preprocess_image(image_path)

    # Classify gaze
    predicted_class, class_probabilities = classify_gaze(model, image_tensor, class_names)

    # Output results
    logging.info(f"Predicted Gaze: {predicted_class}, Probabilities: {class_probabilities}")
        
        
if __name__ == "__main__":
    image_path = "/home/nele_pauline_suffo/ProcessedData/quantex_faces/quantex_at_home_id255237_2022_05_08_01_006060_face_0.jpg"
    main(image_path)
import torch
from fastreid.config import get_cfg
from fastreid.modeling import build_model
from fastreid.utils.checkpoint import Checkpointer
from fastreid.data.transforms import build_transforms
from PIL import Image
import numpy as np

def load_pretrained_model(config_file, model_weights):
    # Set up the configuration
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_weights  # Load pre-trained weights
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Build the model and load weights
    model = build_model(cfg)
    model.eval()
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)
    return model, cfg

def extract_features(model, cfg, image_path):
    # Apply the same transformations as used in training
    transforms = build_transforms(cfg, is_train=False)
    img = Image.open(image_path).convert('RGB')
    img = transforms(img)
    
    # Add batch dimension and move to device
    img = img.unsqueeze(0).to(cfg.MODEL.DEVICE)
    
    # Extract features
    with torch.no_grad():
        features = model(img)
    return features.cpu().numpy()

# Example usage
config_file = 'path_to_config.yaml'  # E.g., "../configs/Market1501/bagtricks_R50.yml"
model_weights = 'path_to_pretrained_weights.pth'  # E.g., "model_final.pth"

model, cfg = load_pretrained_model(config_file, model_weights)
image_path = 'path_to_your_image.jpg'
features = extract_features(model, cfg, image_path)

print("Extracted features:", features)
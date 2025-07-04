import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- 1. Configuration and Hyperparameters ---
class Config:
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define labels for the classification task
    LABELS = [
        "adult_person_present",
        "child_person_present",
        "adult_face_present",
        "child_face_present",
        "object_interaction",
    ]
    NUM_LABELS = len(LABELS)

    DATA_ROOT = "/home/nele_pauline_suffo/ProcessedData/quantex_videos_processed"
    TRAIN_ANNOTATIONS_FILE = "/home/nele_pauline_suffo/ProcessedData/quantex_annotations/cnn_annotations_train.csv"
    VAL_ANNOTATIONS_FILE = "/home/nele_pauline_suffo/ProcessedData/quantex_annotations/cnn_annotations_val.csv"
    TEST_ANNOTATIONS_FILE = "/home/nele_pauline_suffo/ProcessedData/quantex_annotations/cnn_annotations_test.csv"
    
cfg = Config()
print(f"Using device: {cfg.DEVICE}")

# --- 2. Custom Dataset ---
class EgocentricFrameDataset(Dataset):
    def __init__(self, annotations_df, data_root, transform=None):
        self.annotations = annotations_df
        self.data_root = data_root
        self.transform = transform
        
        # Ensure 'frame_id' or similar column exists in your DataFrame
        if 'frame_id' not in self.annotations.columns:
            raise ValueError("Annotations DataFrame must contain a 'frame_id' column.")
        
        # Ensure all labels are present in the DataFrame
        for label in cfg.LABELS:
            if label not in self.annotations.columns:
                raise ValueError(f"Label column '{label}' not found in annotations DataFrame.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        frame_info = self.annotations.iloc[idx]
        frame_id = frame_info['frame_id']
        img_path = os.path.join(self.data_root, f"{frame_id}.jpg") # Assuming .jpg format

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Labels as a torch tensor (multi-hot encoding for multi-label classification)
        labels = torch.FloatTensor([frame_info[label] for label in cfg.LABELS])
        
        return image, labels

# --- 3. Model Architecture ---
class MultiLabelClassifier(nn.Module):
    def __init__(self, num_labels, backbone_name='resnet50', pretrained=True):
        super(MultiLabelClassifier, self).__init__()
        
        # Load pre-trained backbone
        if backbone_name == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            # Remove the original classification head (avgpool and fc layers)
            self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-2]) 
            self.num_features = self.backbone.fc.in_features # This will be 2048 for ResNet50
        elif backbone_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            # EfficientNet features are in backbone.features
            self.feature_extractor = self.backbone.features
            self.num_features = self.backbone.classifier[1].in_features # This is the input feature size for the classifier
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Global Average Pooling to flatten spatial features for the dense layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Multi-head classifier
        self.classification_heads = nn.ModuleDict()
        for i, label_name in enumerate(cfg.LABELS):
            self.classification_heads[label_name] = nn.Sequential(
                nn.Linear(self.num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5), # Add dropout for regularization
                nn.Linear(512, 1),
                nn.Sigmoid() # Binary classification for each head
            )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1) # Flatten for dense layers

        # Apply each classification head
        outputs = {}
        for label_name in cfg.LABELS:
            outputs[label_name] = self.classification_heads[label_name](features).squeeze(1)

        return outputs

# --- 4. Training Function ---
def train_model(model, dataloader, optimizer, criterion, epoch, scheduler=None):
    model.train()
    total_loss = 0
    correct_predictions = {label: 0 for label in cfg.LABELS}
    total_samples = {label: 0 for label in cfg.LABELS}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training")
    for images, labels_tensor in pbar:
        images = images.to(cfg.DEVICE)
        labels_tensor = labels_tensor.to(cfg.DEVICE)

        optimizer.zero_grad()
        outputs = model(images)

        # Calculate loss for each head
        loss = 0
        for i, label_name in enumerate(cfg.LABELS):
            # Ensure target labels match the output shape
            target_labels = labels_tensor[:, i]
            prediction_output = outputs[label_name]
            
            # Binary Cross-Entropy (BCE) expects float targets
            loss += criterion(prediction_output, target_labels)
            
            # For accuracy calculation
            preds = (prediction_output > 0.5).float()
            correct_predictions[label_name] += (preds == target_labels).sum().item()
            total_samples[label_name] += target_labels.size(0)

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        pbar.set_postfix({'batch_loss': loss.item()}) # Display current batch loss

    avg_loss = total_loss / len(dataloader)
    
    accuracies = {label: (correct_predictions[label] / total_samples[label] if total_samples[label] > 0 else 0) * 100 for label in correct_predictions}
    
    print(f"Epoch {epoch} Average Training Loss: {avg_loss:.4f}")
    for label, acc in accuracies.items():
        print(f"  {label} Training Accuracy: {acc:.2f}%")
        
    if scheduler:
        scheduler.step(avg_loss) # Or scheduler.step() for step-based schedulers

# --- 5. Evaluation Function ---
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct_predictions = {label: 0 for label in cfg.LABELS}
    total_samples = {label: 0 for label in cfg.LABELS}

    with torch.no_grad():
        for images, labels_tensor in tqdm(dataloader, desc="Evaluating"):
            images = images.to(cfg.DEVICE)
            labels_tensor = labels_tensor.to(cfg.DEVICE)

            outputs = model(images)
            
            loss = 0
            for i, label_name in enumerate(cfg.LABELS):
                target_labels = labels_tensor[:, i]
                prediction_output = outputs[label_name]
                
                loss += criterion(prediction_output, target_labels)
                
                preds = (prediction_output > 0.5).float()
                correct_predictions[label_name] += (preds == target_labels).sum().item()
                total_samples[label_name] += target_labels.size(0)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    accuracies = {label: (correct_predictions[label] / total_samples[label] if total_samples[label] > 0 else 0) * 100 for label in correct_predictions}

    print(f"Validation Loss: {avg_loss:.4f}")
    for label, acc in accuracies.items():
        print(f"  {label} Validation Accuracy: {acc:.2f}%")
    return avg_loss, accuracies

# --- 6. Main Execution Block ---
if __name__ == "__main__":
    # --- Data Transforms ---
    # ImageNet normalization values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # --- Dataset and DataLoader Setup ---    
    try:
        train_annotations_df = pd.read_csv(cfg.TRAIN_ANNOTATIONS_FILE)
        val_annotations_df = pd.read_csv(cfg.VAL_ANNOTATIONS_FILE)
        test_annotations_df = pd.read_csv(cfg.TEST_ANNOTATIONS_FILE)
    except FileNotFoundError:
        print(f"Error: Annotation file not found at {cfg.TRAIN_ANNOTATIONS_FILE} or {cfg.VAL_ANNOTATIONS_FILE} or {cfg.TEST_ANNOTATIONS_FILE}")
        print("Please create an annotations.csv file and place your video frames in the data_root directory.")
        print("Refer to the comments in the code for the expected CSV format.")
        exit() # Exit if annotations file is not found

    train_dataset = EgocentricFrameDataset(
        annotations_df=train_annotations_df,
        data_root=cfg.DATA_ROOT,
        transform=train_transform
    )
    val_dataset = EgocentricFrameDataset(
        annotations_df=val_annotations_df,
        data_root=cfg.DATA_ROOT,
        transform=val_transform
    )
    test_dataset = EgocentricFrameDataset(
        annotations_df=test_annotations_df,
        data_root=cfg.DATA_ROOT,
        transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2 if os.cpu_count() else 0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2 if os.cpu_count() else 0)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2 if os.cpu_count() else 0)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")


    # --- Model, Loss, Optimizer ---
    model = MultiLabelClassifier(num_labels=cfg.NUM_LABELS, backbone_name='resnet50', pretrained=True)
    model.to(cfg.DEVICE)

    # nn.BCELoss expects sigmoid output from the model (which we have in the heads)
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # --- Training Loop ---
    best_val_loss = float('inf')
    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        train_model(model, train_loader, optimizer, criterion, epoch, scheduler)
        val_loss, _ = evaluate_model(model, val_loader, criterion)

        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
            torch.save(model.state_dict(), "best_multi_label_classifier.pth")
            best_val_loss = val_loss

    print("Training complete!")

    # --- Inference Example ---
    print("\n--- Running Inference Example ---")
    # Load the best model
    model.load_state_dict(torch.load("best_multi_label_classifier.pth"))
    model.eval()

    # Imagine you have a new frame to process
    # For a real system, you'd load frames from your video.
    # Check if DATA_ROOT exists and contains images
    if not os.path.exists(cfg.DATA_ROOT) or not os.listdir(cfg.DATA_ROOT):
        print(f"Error: DATA_ROOT '{cfg.DATA_ROOT}' is empty or does not exist.")
        print("Cannot run inference example without sample images.")
    else:
        dummy_image_path = os.path.join(cfg.DATA_ROOT, os.listdir(cfg.DATA_ROOT)[0]) # Pick a random image for example
        
        try:
            sample_image = Image.open(dummy_image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Could not find a sample image at {dummy_image_path}. Ensure data_root contains images and they are accessible.")
            exit()

        processed_image = val_transform(sample_image).unsqueeze(0).to(cfg.DEVICE) # Add batch dimension

        with torch.no_grad():
            predictions = model(processed_image)

        print("\nInference Results for a sample frame:")
        for label_name in cfg.LABELS:
            prob = predictions[label_name].item()
            print(f"{label_name}: {'Yes' if prob > 0.5 else 'No'} (Probability: {prob:.4f})")
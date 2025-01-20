import os
import torch
import logging
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path
from facenet_pytorch import MTCNN
import torch.optim as optim
import torch.nn as nn
from constants import MtcnnPaths, DetectionPaths

def extract_video_folder(filename):
    """Extract video folder name from image filename."""
    # Remove .jpg extension and split by underscore
    parts = filename.replace('.jpg', '').split('_')
    # Join all parts except the last one (frame number)
    return '_'.join(parts[:-1])

# Dataset Class
class FaceDataset(Dataset):
    def __init__(self, label_file, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.data = []
        
        # Parse label.txt
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                image_name = parts[0]
                video_folder = extract_video_folder(image_name)
                bbox = list(map(float, parts[1].split(',')))  # x1, y1, width, height
                x1, y1, width, height = bbox[:4]
                x2, y2 = x1 + width, y1 + height
                self.data.append({
                    'image_path': image_name,
                    'video_folder': video_folder,
                    'bbox': [x1, y1, x2, y2]
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        original_idx = idx
        while idx < len(self.data):
            item = self.data[idx]
            image_path = self.image_dir / item['video_folder'] / item['image_path']
            
            if image_path.exists():
                print(f"Loading image: {image_path}")
                image = Image.open(str(image_path)).convert('RGB')
                bbox = torch.tensor(item['bbox'], dtype=torch.float32)
                
                if self.transform:
                    image = self.transform(image)
                
                return image, bbox
            else:
                logging.warning(f"Skipping missing image: {image_path}")
                idx += 1

# EarlyStopping Class
class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            logging.info(f"No improvement in validation loss. Counter: {self.counter}/{self.patience}")
            return self.counter >= self.patience

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, patience=5):
    model.train()
    early_stopping = EarlyStopping(patience)
    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        for images, bboxes in train_loader:
            optimizer.zero_grad()
            _, _, output_bboxes = model.detect(images, landmarks=False)
            output_bboxes = torch.tensor(output_bboxes, dtype=torch.float32)
            
            # Compute loss
            loss = criterion(output_bboxes, bboxes)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation loss
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, bboxes in val_loader:
                _, _, output_bboxes = model.detect(images, landmarks=False)
                output_bboxes = torch.tensor(output_bboxes, dtype=torch.float32)
                loss = criterion(output_bboxes, bboxes)
                val_loss += loss.item()
        
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if early_stopping.step(val_loss):
            logging.info("Early stopping triggered.")
            break

# Main Function
def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((160, 160)),  # Resize to MTCNN input size
        transforms.ToTensor()
    ])
    
    # Dataset and DataLoader setup
    dataset = FaceDataset(MtcnnPaths.labels_file_path, DetectionPaths.images_input_dir, transform)
    
    # Split the dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pretrained MTCNN model
    mtcnn = MTCNN(
        image_size=224,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        keep_all=True,
        device=device,
        )
    mtcnn.eval()

    # Fine-tuning components
    optimizer = optim.Adam(mtcnn.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Train the model
    train_model(mtcnn, train_loader, val_loader, criterion, optimizer, num_epochs=10)
    
    # Save the fine-tuned model
    torch.save(mtcnn.state_dict(), f"{MtcnnPaths.output_dir}/fine_tuned_mtcnn.pth")

# Run the main function
if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from src.projects.social_interactions.common.constants import ResNetParameters as RNP, DetectionPaths
from src.projects.social_interactions.config.config import ResNetConfig as RNC, TrainingConfig
#from torchvision.models import ResNet18_Weights

class GazeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        This class is used to load the dataset for the gaze estimation model.
        
        Parameters
        ----------
        csv_file : Path
            Path to the CSV file with image names and labels.
        root_dir : Path
            Path to the directory with the images.
        transform : callable, optional
        """
        # Load the labels from the CSV file
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        # Load the image and label
        img_name = self.root_dir / self.labels_df.iloc[idx, 0]
        image = Image.open(img_name).convert("RGB")
        label = self.labels_df.iloc[idx, 1]
        label = torch.tensor(label, dtype=torch.long)

        return image, label

# class GazeEstimationModel(nn.Module):
#     def __init__(self, pretrained=True):
#         super(GazeEstimationModel, self).__init__()
#         # Determine the weights based on the pretrained argument
#         weights = ResNet18_Weights.DEFAULT if pretrained else None
#         # Load a pretrained ResNet-18 model with the new weights format
#         self.backbone = models.resnet18(weights=weights)
#         # Change the last layer to output a single value
#         self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
#         # Sigmoid activation function to output a value between 0 and 1
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # Forward pass through the model
#         x = self.backbone(x)
#         return self.sigmoid(x)

def train_model():
    # Define the transformations for the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the dataset
    dataset = GazeDataset(csv_file=RNP.gaze_labels_csv_path, root_dir=DetectionPaths.images_input_dir, transform=transform)
    
     # Set random seed for reproducibility
    random_seed = TrainingConfig.random_seed
    torch.manual_seed(random_seed)
    random.seed(random_seed)

     # Split the dataset into training and validation sets
    train_size = int(TrainingConfig.train_test_split_ratio * len(dataset))
    val_size = len(dataset) - train_size

    # Create a generator and set the seed for reproducibility
    generator = torch.Generator().manual_seed(random_seed)

    # Perform the split with the generator
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    # Create DataLoaders for each subset
    train_loader = DataLoader(train_dataset, batch_size=RNC.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=RNC.batch_size, shuffle=False, num_workers=4)
    
    # Load the pretrained model
    model = GazeEstimationModel(pretrained=True).to('cuda:0')
    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    num_epochs = RNC.num_epochs

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to('cuda:0'), labels.float().unsqueeze(1).to('cuda:0')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to('cuda:0'), labels.float().unsqueeze(1).to('cuda:0')
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            print(f'Validation Loss: {val_loss/len(val_loader)}, Accuracy: {100 * correct / total}%')

    torch.save(model.state_dict(), RNP.trained_model_path)

if __name__ == "__main__":
    train_model()

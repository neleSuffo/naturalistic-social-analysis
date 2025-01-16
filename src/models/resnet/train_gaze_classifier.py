import torch
import logging
import os
import cv2
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torch.optim import Adam
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from constants import ResNetPaths
from collections import Counter

cv2.setNumThreads(2)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Define the dataset class
class GazeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Function to read data
def read_data(file_path):
    image_paths = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            image_paths.append(parts[0])
            labels.append(int(parts[1]))
    return image_paths, labels

# Function to perform stratified split
def stratified_split(image_paths, labels, train_size=0.8, val_size=0.1, test_size=0.1):
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=(1 - train_size), random_state=42, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=(test_size / (test_size + val_size)), random_state=42, stratify=temp_labels
    )
    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels

# Function to create transformations
def get_transformations():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Function to create the model
def create_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Freeze all layers except the final fully connected layer
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)  # Modify for binary classification
    return model

# Function to calculate recall
def calculate_recall(all_targets, all_predictions):
    return recall_score(all_targets, all_predictions, average='macro')

# Function to plot confusion matrix
def plot_confusion_matrix(cm, class_names, output_dir, epoch=None):
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    
    if epoch is not None:
        plt.savefig(f"{output_dir}/confusion_matrix_epoch_{epoch}.png")
    else:
        plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()

# Function to save and plot other metrics
def save_and_plot_metrics(epoch, train_loss, train_recall, val_loss, val_recall, output_dir):
    # Save the loss and recall curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{output_dir}/loss_curve.png")
    
    plt.subplot(1, 2, 2)
    plt.plot(train_recall, label='Train Recall')
    plt.plot(val_recall, label='Val Recall')
    plt.title('Recall Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()
    plt.savefig(f"{output_dir}/recall_curve.png")
    plt.close()

# Define the FocalLoss class
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=-1)
        p_t = inputs.gather(1, targets.view(-1, 1))  # Get probability for the true class
        loss = -self.alpha * (1 - p_t) ** self.gamma * torch.log(p_t)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, device, output_dir, patience=5):
    best_val_recall = 0.0
    patience_counter = 0
    num_epochs = 40
    train_loss, train_recall, val_loss, val_recall = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_targets = []
        all_predictions = []

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Store predictions and targets for recall calculation
            _, predicted = torch.max(outputs, 1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            running_loss += loss.item()

        epoch_recall = calculate_recall(all_targets, all_predictions)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Recall: {epoch_recall:.4f}")
        
        train_loss.append(running_loss / len(train_loader))
        train_recall.append(epoch_recall)

        # Validate on the validation set
        model.eval()
        val_loss_epoch = 0.0
        val_all_targets = []
        val_all_predictions = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss_epoch += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_all_targets.extend(targets.cpu().numpy())
                val_all_predictions.extend(predicted.cpu().numpy())

        val_recall_epoch = calculate_recall(val_all_targets, val_all_predictions)
        logger.info(f"Validation Loss: {val_loss_epoch / len(val_loader):.4f}, Validation Recall: {val_recall_epoch:.4f}")

        val_loss.append(val_loss_epoch / len(val_loader))
        val_recall.append(val_recall_epoch)

        # Plot and save confusion matrix
        val_cm = confusion_matrix(val_all_targets, val_all_predictions)
        plot_confusion_matrix(val_cm, class_names=['No Gaze', 'Gaze'], output_dir=output_dir, epoch=epoch)

        # Save metrics plots
        save_and_plot_metrics(epoch, train_loss, train_recall, val_loss, val_recall, output_dir)

        if val_recall_epoch > best_val_recall:
            best_val_recall = val_recall_epoch
            patience_counter = 0
            torch.save(model.state_dict(), f"{output_dir}/best_gaze_classification_model.pth")
            logger.info("Validation recall improved. Model saved.")
        else:
            patience_counter += 1
            logger.info(f"No improvement in validation recall. Patience counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    return model

# Function to evaluate the model
def evaluate_model(model, test_loader, criterion, device, output_dir):
    model.eval()
    test_loss = 0.0
    test_all_targets = []
    test_all_predictions = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            test_all_targets.extend(targets.cpu().numpy())
            test_all_predictions.extend(predicted.cpu().numpy())

    test_recall = calculate_recall(test_all_targets, test_all_predictions)
    logger.info(f"Test Loss: {test_loss / len(test_loader):.4f}, Test Recall: {test_recall:.4f}")

    test_cm = confusion_matrix(test_all_targets, test_all_predictions)
    plot_confusion_matrix(test_cm, class_names=['No Gaze', 'Gaze'], output_dir=output_dir)

    # Plot and save ROC and Precision-Recall curves
    fpr, tpr, _ = roc_curve(test_all_targets, test_all_predictions)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(f"{output_dir}/roc_curve.png")
    plt.close()

    precision, recall, _ = precision_recall_curve(test_all_targets, test_all_predictions)
    average_precision = average_precision_score(test_all_targets, test_all_predictions)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig(f"{output_dir}/precision_recall_curve.png")
    plt.close()

# Main function
def main():
    # Specify the output directory for saving plots
    output_dir = ResNetPaths.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Read the data
    image_paths, labels = read_data('/home/nele_pauline_suffo/ProcessedData/quantex_faces/face_labels.txt')

    # Stratified split into train, validation, and test sets
    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = stratified_split(image_paths, labels)

    # Define data transformations
    transform = get_transformations()

    # Create datasets
    train_dataset = GazeDataset(train_paths, train_labels, transform=transform)
    val_dataset = GazeDataset(val_paths, val_labels, transform=transform)
    test_dataset = GazeDataset(test_paths, test_labels, transform=transform)

    # Handle class imbalance with a weighted sampler
    class_counts = Counter(labels)
    class_weights = [1.0 / class_counts[c] for c in range(len(class_counts))]
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Create model
    model = create_model()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define criterion (Focal Loss) and optimizer
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = Adam(model.fc.parameters(), lr=0.001, weight_decay=1e-4)

    # Train the model
    model = train_model(model, train_loader, val_loader, criterion, optimizer, device, output_dir)

    # Evaluate on the test set
    evaluate_model(model, test_loader, criterion, device, output_dir)

if __name__ == "__main__":
    main()
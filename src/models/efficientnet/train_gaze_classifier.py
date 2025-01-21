import logging
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, precision_score, recall_score, roc_auc_score, accuracy_score
import os
import random
import shutil
from constants import EfficientNetPaths, MtcnnPaths

logging.basicConfig(level=logging.INFO)

class EnhancedEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(EnhancedEfficientNet, self).__init__()
        # Load pre-trained EfficientNet
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b3')
        # Get the number of features from the EfficientNet model
        in_features = self.efficientnet._fc.in_features 

        # Define a new classifier with more capacity
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),  # Additional layer
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),  # Another additional layer
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, num_classes)  # Output layer
        )

    def forward(self, x):
        # Extract features
        features = self.efficientnet.extract_features(x)
        # Global average pooling
        x = F.adaptive_avg_pool2d(features, output_size=1)
        # Flatten
        x = x.view(x.size(0), -1)
        # Classification
        x = self.classifier(x)
        return x
    
class FocalLoss(nn.Module):
    """
    Focal Loss for binary/multi-class classification.
    Args:
        alpha (float): Balances the importance of positive/negative classes (default=0.25).
        gamma (float): Modulates the importance of well-classified examples (default=2.0).
        reduction (str): Specifies the reduction to apply to the output ('none', 'mean', or 'sum').
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)  # Probabilities of the predicted classes
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
def organize_data(source_dir, labels_file, train_dir, val_dir, test_dir, train_ratio=0.8, val_ratio=0.1):
    """
    This function organizes the data into train, validation, and test sets.
    
    Parameters:
        source_dir (str): Directory containing the source images.
        labels_file (str): Path to the file containing image paths and labels.
        train_dir (str): Directory to save the training set.
        val_dir (str): Directory to save the validation set.
        test_dir (str): Directory to save the test set.
        train_ratio (float): Ratio of training data (default=0.8).
        val_ratio (float): Ratio of validation data (default=0.1).
    
    Returns:
        None
    """
    # Check if data is already organized
    if all(os.path.exists(os.path.join(split_dir, '0')) and 
        os.path.exists(os.path.join(split_dir, '1')) and 
        os.listdir(os.path.join(split_dir, '0')) and 
        os.listdir(os.path.join(split_dir, '1')) 
        for split_dir in [train_dir, val_dir, test_dir]):
            logging.info("Train, validation, and test subfolders already exist and are not empty. Skipping data organization.")
            return
    
    # Create directories for classes
    for split_dir in [train_dir, val_dir, test_dir]:
        for class_idx in ['0', '1']:
            os.makedirs(os.path.join(split_dir, class_idx), exist_ok=True)
    
    # Read labels
    with open(labels_file, 'r') as f:
        lines = f.readlines()
    
    # Create image-label pairs with absolute paths
    image_labels = []
    for line in lines:
        full_img_path, label = line.strip().split()
        # Verify source image exists
        if os.path.exists(full_img_path):
            image_labels.append((full_img_path, label))
        else:
            logging.warning(f"Image not found: {full_img_path}")
    
    # Shuffle and split
    random.shuffle(image_labels)
    total = len(image_labels)
    train_idx = int(total * train_ratio)
    val_idx = int(total * (train_ratio + val_ratio))
    
    # Split data
    train_data = image_labels[:train_idx]
    val_data = image_labels[train_idx:val_idx]
    test_data = image_labels[val_idx:]
    
    # Copy files to appropriate directories
    for data, split_dir in [(train_data, train_dir), 
                           (val_data, val_dir), 
                           (test_data, test_dir)]:
        for src_path, label in data:
            # Create destination path using just the filename
            filename = os.path.basename(src_path)
            dst_dir = os.path.join(split_dir, label)
            dst_path = os.path.join(dst_dir, filename)
            
            try:
                shutil.copy(src_path, dst_path)
            except Exception as e:
                logging.error(f"Failed to copy {src_path} to {dst_path}: {e}")
    
    logging.info(f"Split completed:\nTrain: {len(train_data)}\nVal: {len(val_data)}\nTest: {len(test_data)}")
   
 
def create_data_loaders(train_dir, val_dir, test_dir, image_size=(150, 150), batch_size=32):
    """
    Create data loaders for train, validation, and test sets.
    """
    transform_train = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_test)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)
    
    # Calculate class weights based on the dataset
    class_counts = [0] * len(train_dataset.classes)
    for _, label in train_dataset.imgs:
        class_counts[label] += 1
    class_weights = [1.0 / count for count in class_counts]

    # Assign a weight to each sample in the training dataset
    sample_weights = [class_weights[label] for _, label in train_dataset.imgs]

    # Create the WeightedRandomSampler
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader

def prepare_environment(source_dir, labels_file, train_dir, val_dir, test_dir, base_dir=EfficientNetPaths.output_dir):
    """
    This function prepares the environment for training the model.
    
    Parameters:
        source_dir (str): Directory containing the source images.
        labels_file (str): Path to the file containing image paths and labels.
        train_dir (str): Directory to save the training set.
        val_dir (str): Directory to save the validation set.
        test_dir (str): Directory to save the test set.
        base_dir (str): Base directory to save the results (default='output').
        
    Returns:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        output_dir (str): Directory to save the results.
    """
    # Organize data into class folders
    organize_data(source_dir, labels_file, train_dir, val_dir, test_dir)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(train_dir, val_dir, test_dir)

    # Create output directory for results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy the script to the output directory
    script_path = __file__
    shutil.copy(script_path, os.path.join(output_dir, os.path.basename(script_path)))

    return train_loader, val_loader, test_loader, output_dir

def train_model(train_loader, val_loader, num_epochs, num_classes=2, device='cuda'):
    """
    Train a model using the train and validation loaders.
    """
    model = EnhancedEfficientNet(num_classes=num_classes)
    model = model.to(device)
    
    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    best_val_loss = float('inf')
    patience = 5
    no_improve_count = 0

    train_losses = []
    val_losses = []
    val_recalls = []  # Store validation recall per epoch

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        all_true_labels = []
        all_predictions = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_true_labels.extend(labels.cpu().numpy())
                all_predictions.extend(preds.cpu().numpy())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Calculate recall for the validation set
        val_recall = recall_score(all_true_labels, all_predictions, average='macro')
        val_recalls.append(val_recall)

        logging.info(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Recall: {val_recall:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            no_improve_count = 0
        else:
            no_improve_count += 1
            logging.info(f"No improvement for {no_improve_count} epochs. Best val_loss: {best_val_loss:.4f}")
        if no_improve_count >= patience:
            logging.info("Early stopping triggered.")
            break

        scheduler.step()

    return best_model, train_losses, val_losses, val_recalls


def evaluate_and_plot_results(model, test_loader, class_names, output_dir, device='cuda'):
    """
    Evaluate the model on the test set, compute metrics, and save evaluation plots.
    """
    logging.info("Evaluating model and generating plots...")

    # Evaluate the model
    model.eval()
    true_labels = []
    predictions = []
    probabilities = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)

            true_labels.extend(labels.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    
    # Save test metrics
    save_test_metrics(true_labels, predictions, probabilities,class_names, output_dir)

    # Generate plots
    confusion = confusion_matrix(true_labels, predictions)
    save_confusion_matrix(confusion, class_names, output_dir)
    save_precision_recall_curve(true_labels, probabilities, class_names, output_dir)
    save_roc_curve(true_labels, probabilities, class_names, output_dir)

    logging.info(f"Evaluation and plotting completed. Results saved to {output_dir}")

def save_precision_recall_curve(true_labels, predictions, class_names, output_dir):
    """
    Save the precision-recall curve for each class.
    """
    true_one_hot = np.eye(len(class_names))[true_labels]
    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(true_one_hot[:, i], predictions[:, i])
        avg_precision = average_precision_score(true_one_hot[:, i], predictions[:, i])
        plt.plot(recall, precision, label=f'{class_name} (AP = {avg_precision:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plot_path = os.path.join(output_dir, "precision_recall_curve.png")
    plt.savefig(plot_path)
    plt.close()
    
def save_confusion_matrix(confusion, class_names, output_dir):
    """
    Save the confusion matrix plot.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            plt.text(j, i, confusion[i, j],
                     horizontalalignment="center",
                     color="white" if confusion[i, j] > confusion.max() / 2 else "black")

    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(plot_path)
    logging.info(f"Confusion matrix saved to {plot_path}")
    plt.close()

def save_roc_curve(true_labels, predictions, class_names, output_dir):
    """
    Save the ROC curve plot.
    """
    true_one_hot = np.eye(len(class_names))[true_labels]
    fpr, tpr, _ = roc_curve(true_one_hot.ravel(), predictions.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plot_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(plot_path)
    logging.info(f"ROC curve saved to {plot_path}")
    plt.close()

def save_loss_plot(train_losses, val_losses, output_dir):
    """
    Save the training and validation loss plot.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(plot_path)
    logging.info(f"Loss curve saved to {plot_path}")
    plt.close()

def save_recall_plot(val_recalls, output_dir):
    """
    Save the validation recall plot.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(val_recalls, label='Validation Recall', color='green')
    plt.title("Validation Recall")
    plt.xlabel("Epochs")
    plt.ylabel("Recall")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_dir, "recall_curve.png")
    plt.savefig(plot_path)
    logging.info(f"Recall curve saved to {plot_path}")
    plt.close()
    
def save_test_metrics(true_labels, predictions, probabilities, class_names, output_dir):
    """
    Save test metrics to a JSON file.
    
    Args:
        true_labels (np.ndarray): Ground truth labels
        predictions (np.ndarray): Model predictions
        probabilities (np.ndarray): Prediction probabilities
        class_names (list): List of class names
        output_dir (str): Directory to save metrics
    """
    try:
        # Calculate overall metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='macro')
        recall = recall_score(true_labels, predictions, average='macro')
        roc_auc = roc_auc_score(true_labels, probabilities[:, 1])  # For binary classification
        avg_precision = average_precision_score(true_labels, probabilities[:, 1])
        
        # Calculate per-class recall
        per_class_recall = recall_score(true_labels, predictions, average=None)
        per_class_recall_dict = {
            class_names[i]: float(per_class_recall[i])
            for i in range(len(class_names))
        }
        
        # Prepare metrics dictionary
        metrics = {
            "Test Accuracy": float(accuracy),
            "Test Precision": float(precision),
            "Test Recall (Average)": float(recall),
            "AUC-ROC": float(roc_auc),
            "Average Precision": float(avg_precision),
            "Number of Test Samples": len(true_labels),
            "Class Distribution": {
                class_names[0]: int(np.sum(true_labels == 0)),
                class_names[1]: int(np.sum(true_labels == 1)),
            },
            "Per-Class Recall": per_class_recall_dict,
        }
        
        # Save metrics to a JSON file
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, "test_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Test metrics saved to {metrics_path}")
        
    except Exception as e:
        logging.error(f"Failed to save test metrics: {str(e)}")
         
def main():
    # Set directories
    source_dir = MtcnnPaths.faces_dir
    train_dir = os.path.join(MtcnnPaths.faces_dir, 'train')
    val_dir = os.path.join(MtcnnPaths.faces_dir, 'val')
    test_dir = os.path.join(MtcnnPaths.faces_dir, 'test')
    labels_file = MtcnnPaths.gaze_labels_file_path

    # Prepare data and environment
    train_loader, val_loader, test_loader, output_dir = prepare_environment(
        source_dir, labels_file, train_dir, val_dir, test_dir
    )
    
    # Train the model
    model, train_losses, val_losses, val_recalls = train_model(train_loader, val_loader, num_epochs=50)
    
    # Save training and validation plots
    save_loss_plot(train_losses, val_losses, output_dir)
    save_recall_plot(val_recalls, output_dir)

    # Get class names
    class_names = os.listdir(train_dir)

    # Evaluate and generate plots
    evaluate_and_plot_results(model, test_loader, class_names, output_dir)

if __name__ == '__main__':
    main()
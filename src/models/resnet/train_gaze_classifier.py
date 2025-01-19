import logging
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, precision_score, recall_score
import os
import random
import shutil
from constants import EfficientNetPaths, MtcnnPaths

logging.basicConfig(level=logging.INFO)

def create_output_directory(base_dir=EfficientNetPaths.output_dir):
    """
    Create a timestamped output directory.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_current_script(output_dir):
    """
    Save a copy of the current script to the output directory.
    """
    script_path = __file__
    shutil.copy(script_path, os.path.join(output_dir, os.path.basename(script_path)))
    
def organize_data(source_dir, labels_file, train_dir, val_dir, test_dir, train_ratio=0.8, val_ratio=0.1):
    """Organize images into class folders and split into train/val/test"""
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader


def train_model(train_loader, val_loader, num_epochs, num_classes=2, device='cuda'):
    """
    Train a model using the train and validation loaders.
    """
    model = models.efficientnet_b3(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
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

    # Compute metrics
    accuracy = np.mean(true_labels == predictions)
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')

    # Save test metrics
    save_test_metrics(accuracy, precision, recall, output_dir)

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
    
def save_test_metrics(accuracy, precision, recall, output_dir):
    """
    Save test metrics to a JSON file.
    """
    metrics = {
        "Test Accuracy": accuracy,
        "Test Precision": precision,
        "Test Recall": recall,
        "AUC-ROC": roc_auc,
        "Average Precision": average_precision_score(true_labels, probabilities, average='macro')
    }
    metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Test metrics saved to {metrics_path}")
   
def prepare_environment(source_dir, labels_file, train_dir, val_dir, test_dir):
    """
    Prepare data and create output directories.
    """
    # Organize data into class folders
    organize_data(source_dir, labels_file, train_dir, val_dir, test_dir)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(train_dir, val_dir, test_dir)

    # Create output directory for results
    output_dir = create_output_directory()
    save_current_script(output_dir)

    return train_loader, val_loader, test_loader, output_dir
         
def main():
    # Set directories
    source_dir = MtcnnPaths.faces_dir
    train_dir = os.path.join(MtcnnPaths.faces_dir, 'train')
    val_dir = os.path.join(MtcnnPaths.faces_dir, 'val')
    test_dir = os.path.join(MtcnnPaths.faces_dir, 'test')
    labels_file = MtcnnPaths.face_labels_file_path

    # Prepare data and environment
    train_loader, val_loader, test_loader, output_dir = prepare_environment(
        source_dir, labels_file, train_dir, val_dir, test_dir
    )
    
    # Train the model
    model, train_losses, val_losses, val_recalls = train_model(train_loader, val_loader, num_epochs=5)
    
    # Save training and validation plots
    save_loss_plot(train_losses, val_losses, output_dir)
    save_recall_plot(val_recalls, output_dir)

    # Get class names
    class_names = os.listdir(train_dir)

    # Evaluate and generate plots
    evaluate_and_plot_results(model, test_loader, class_names, output_dir)

if __name__ == '__main__':
    main()
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os
import random
import shutil
from math import ceil, sqrt
from constants import EfficientNetPaths, MtcnnPaths

logging.basicConfig(level=logging.INFO)


def organize_data(source_dir, labels_file, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.15):
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


def train_model(train_loader, val_loader, num_classes=2, device='cuda'):
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

    for epoch in range(50):
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

        model.eval()
        val_loss = 0
        val_recall = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_recall += (preds == labels).sum().item() / len(labels)

        val_loss /= len(val_loader)
        val_recall /= len(val_loader)

        logging.info(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Recall: {val_recall:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            logging.info("Early stopping triggered.")
            break

        scheduler.step()

    return best_model


def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate the model on the test set.
    """
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.numpy())

    return np.array(true_labels), np.array(predictions)


def plot_confusion_matrix(true_labels, predictions, class_names):
    """
    Plot the confusion matrix.
    """
    confusion = confusion_matrix(true_labels, predictions)
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
    plt.show()


def plot_roc_curve(true_labels, predictions, class_names):
    """
    Plot the ROC curve.
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
    plt.show()


def main():
    # Set directories
    source_dir = MtcnnPaths.faces_dir
    train_dir = os.path.join(MtcnnPaths.faces_dir, 'train')
    val_dir = os.path.join(MtcnnPaths.faces_dir, 'val')
    test_dir = os.path.join(MtcnnPaths.faces_dir, 'test')
    labels_file = MtcnnPaths.face_labels_file_path

    # Organize data into class folders
    organize_data(source_dir, labels_file, train_dir, val_dir, test_dir)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(train_dir, val_dir, test_dir)

    model = train_model(train_loader, val_loader)

    true_labels, predictions = evaluate_model(model, test_loader)
    class_names = os.listdir(train_dir)

    plot_confusion_matrix(true_labels, predictions, class_names)

if __name__ == '__main__':
    main()
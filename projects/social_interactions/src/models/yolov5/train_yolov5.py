import yaml
import torch
import sys
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from projects.social_interactions.src.common.constants import YoloParameters, DetectionPaths
from shared.utils import fetch_all_annotations, load_yolo_model
from shared.video_frame_dataset import VideoFrameDataset

sys.path.append('/Users/nelesuffo/projects/leuphana-IPE/yolov5/')
from yolov5.utils.loss import ComputeLoss
from yolov5.utils.torch_utils import ModelEMA


# Define transforms
transform = transforms.Compose([
    transforms.ToTensor()
])

# Fetch all annotations
annotations = fetch_all_annotations(DetectionPaths.annotations_db_path)

# Create dataset (annotations need to stay ordered by frame to load the video frames correctly)
# Frame, bbox, category_id
dataset = VideoFrameDataset(annotations, transform=transform)

# Generate indices for the training and validation sets
indices = list(range(len(dataset)))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

# Create subset datasets
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# Create train and validation data loaders
train_loader = DataLoader(train_dataset, batch_size=YoloParameters.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=YoloParameters.batch_size, shuffle=False, num_workers=4)

# Load hyperparameters
with open(YoloParameters.hyp_path, 'r') as file:
    hyp = yaml.safe_load(file)

# Load model
# <class 'models.common.AutoShape'>
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_yolo_model()

# Configure optimizer
nbs = 64  # nominal batch size
accumulate = max(round(nbs / YoloParameters.batch_size), 1)
hyp['weight_decay'] *= YoloParameters.batch_size * accumulate / nbs
optimizer = torch.optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

# EMA
ema = ModelEMA(model)

# Loss function
compute_loss = ComputeLoss(model)

# Scheduler
lf = lambda x: (1 - x / YoloParameters.epochs) * (1.0 - hyp['lrf']) + hyp['lrf']
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

# Training loop
for epoch in range(YoloParameters.epochs):
    model.train()
    mloss = torch.zeros(4, device=device)  # mean losses

    for i, (imgs, targets, _) in enumerate(train_loader):
        imgs = imgs.to(device)
        targets = targets.to(device)

        # Forward
        pred = model(imgs)
        
        # Compute loss
        loss, loss_items = compute_loss(pred, targets)
        
        # Backward
        loss.backward()
        
        # Optimize
        optimizer.step()
        optimizer.zero_grad()
        
        # EMA
        ema.update(model)
        
        # Print statistics
        mloss = (mloss * i + loss_items) / (i + 1)
    
    scheduler.step()

    # Validation step
    model.eval()
    for i, (imgs, targets, _) in enumerate(val_loader):
        imgs = imgs.to(device)
        targets = targets.to(device)

        # Forward
        with torch.no_grad():
            pred = model(imgs)

        # Compute loss
        loss, loss_items = compute_loss(pred, targets)

    print(f"Epoch {epoch + 1}/{YoloParameters.epochs}, Loss: {mloss}")

# Save the final model
torch.save(ema.ema.state_dict(), 'best.pt')

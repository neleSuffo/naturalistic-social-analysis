import sys
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from common.constants import ANNOTATIONS_DB_PATH, BATCH_SIZE, ANNOTATIONS_DB_PATH, HYP_PATH, PRETRAINED_WEIGHTS_PATH, DATA_CONFIG_PATH
from shared.video_frame_dataset import VideoFrameDataset
from shared.utils import fetch_all_annotations
from yolov5 import train  # Import YOLOv5 training script

# Ensure the YOLOv5 repository is in your PYTHONPATH
sys.path.append('path_to_yolov5_directory')  # Update this to your YOLOv5 directory

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor()
])

# Fetch all annotations
annotations = fetch_all_annotations(ANNOTATIONS_DB_PATH)

# Split annotations
train_annotations, val_annotations = train_test_split(annotations, test_size=0.2, random_state=42)

# Create datasets
train_dataset = VideoFrameDataset(train_annotations, transform=transform)
val_dataset = VideoFrameDataset(val_annotations, transform=transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# Paths to pre-trained weights and other configurations
pretrained_weights_path = PRETRAINED_WEIGHTS_PATH 
data_config_path = DATA_CONFIG_PATH

# Training configuration
with open(HYP_PATH, 'r') as file:
    hyp = yaml.safe_load(file)

# Train YOLOv5 model
if __name__ == '__main__':
    train.run(
        data=data_config_path,  # path to data.yaml
        imgsz=640,  # image size
        batch_size=BATCH_SIZE,  # batch size
        epochs=50,  # number of epochs
        hyp=hyp,  # hyperparameters
        weights=pretrained_weights_path,  # path to pre-trained weights
        project='runs/train',  # project name
        name='exp',  # experiment name
        exist_ok=True  # whether to overwrite existing experiment
    )

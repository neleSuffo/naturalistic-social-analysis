import sys
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from projects.social_interactions.src.common.constants import YoloParameters, DetectionPaths
from shared.video_frame_dataset import VideoFrameDataset
from shared.utils import fetch_all_annotations
from yolov5 import train as yolo_train  # Import YOLOv5 training script
from projects.social_interactions.src.models.yolov5.utils.utils import load_yolo_model

# Ensure the YOLOv5 repository is in your PYTHONPATH
sys.path.append('yolov5')  

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor()
])

# Fetch all annotations
annotations = fetch_all_annotations(DetectionPaths.annotations_db_path)

# Split annotations into training and validation sets
train_annotations, val_annotations = train_test_split(annotations, test_size=0.2, random_state=42)

# Create train and validation datasets
train_dataset = VideoFrameDataset(train_annotations, transform=transform)
val_dataset = VideoFrameDataset(val_annotations, transform=transform)

# Create train and validation data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# Load YOLOv5 model
yolo_model = load_yolo_model()

# Training configuration
with open(YoloParameters.hyp_path, 'r') as file:
    hyp = yaml.safe_load(file)

# Train YOLOv5 model    
if __name__ == '__main__':
    yolo_train.run(
        data=YoloParameters.data_config_path,  # path to data.yaml
        imgsz=640,  # image size
        batch_size=YoloParameters.batch_size,  # batch size
        epochs=50,  # number of epochs
        hyp=hyp,  # hyperparameters
        weights=YoloParameters.pretrained_weights_path,  # path to pre-trained weights
        project='runs/train',  # project name
        name='exp',  # experiment name
        exist_ok=True  # whether to overwrite existing experiment
    )

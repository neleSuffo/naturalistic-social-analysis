import os
import subprocess
from projects.social_interactions.src.common.constants import YoloParameters


# Construct the full path to the train.py script in the YOLOv5 repository
train_script_path = os.path.join(YoloParameters.yolov5_repo_path, 'train.py')

# Construct the training command
train_cmd = [
    'python', train_script_path,
    '--img', str(YoloParameters.img_size),
    '--batch', str(YoloParameters.batch_size),
    '--epochs', str(YoloParameters.epochs),
    '--data', YoloParameters.data_config_path,
    '--project', 'projects/social_interactions/src/outputs/yolov5/train',
    '--name', 'exp',
]

if __name__ == "__main__":
    # Run the training command
    subprocess.run(train_cmd)

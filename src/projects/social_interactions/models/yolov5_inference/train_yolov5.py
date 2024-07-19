import os
import subprocess
from src.projects.social_interactions.common.constants import YoloParameters as Yolo

# Construct the full path to the train.py script in the YOLOv5 repository
train_script_path = os.path.join(Yolo.yolov5_repo_path, "train.py")

# Construct the training command
train_cmd = [
    "python",
    train_script_path,
    "--img",
    str(Yolo.img_size),
    "--batch",
    str(Yolo.batch_size),
    "--epochs",
    str(Yolo.epochs),
    "--data",
    str(Yolo.data_config),
    "--project",
    "src/projects/social_interactions/outputs/yolov5/train",
    "--name",
    "exp",
]

if __name__ == "__main__":
    # Run the training command
    subprocess.run(train_cmd)

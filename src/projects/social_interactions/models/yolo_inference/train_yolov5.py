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
    "outputs/yolov5/train",
    "--name",
    "exp",
    "--device",
    "0,1"  # Specify both GPUs (GPU 0 and GPU 1)
]
if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '10'
    # Run the training command
    subprocess.run(train_cmd)
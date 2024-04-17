import supervision as sv
import numpy as np
from ultralytics import YOLO

# load data 
#video_path = "data/sample_1_short.MP4"
#video_path = "data/patrol_eyes_old.MP4"
video_path = "data/sample_2_short.MP4"


# load model for inference
model = YOLO("yolov8x.pt")

results = model(video_path, save=True)

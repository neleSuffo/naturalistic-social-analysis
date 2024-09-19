import cv2
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Add leuphana_ipe directory to the system path
sys.path.append('/home/nele_pauline_suffo/projects/boxmot')
from boxmot import DeepOCSORT

yolo_model = YOLO('/home/nele_pauline_suffo/models/yolov8_trained.pt')


tracker = DeepOCSORT(
    model_weights=Path('resnet50_dukemtmcreid.pt'), # which ReID model to use
    device='cuda:0',
    fp16=False,
)

# Define the path to your video file
video_path = '/home/nele_pauline_suffo/ProcessedData/videos/quantex_at_home_id254922_2022_04_12_01.MP4'

# Open the video file
video = cv2.VideoCapture(video_path)
                       
while True:
    ret, image = video.read()

    # Run the YOLO model
    results = yolo_model(image)  
    
    #Convert the results to a pandas DataFrame
    detection = results[0].boxes
    cls = detection.cls.cpu().numpy() if torch.is_tensor(detection.cls) else detection.cls
    conf = detection.conf.cpu().numpy() if torch.is_tensor(detection.conf) else detection.conf
    xyxy = detection.xyxy.cpu().numpy() if torch.is_tensor(detection.xyxy) else detection.xyxy

    # Stack the extracted attributes into a NumPy array of shape N x (x, y, x, y, conf, cls)
    formatted_output = np.column_stack((xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3], conf, cls))

    # Check if there are any detections
    if formatted_output.size > 0:
        tracker.update(formatted_output, image) # --> M X (x, y, x, y, id, conf, cls, ind)
    # If no detections, make prediction ahead
    else:   
        formatted_output = np.empty((0, 6))  # empty N X (x, y, x, y, conf, cls)
        tracker.update(formatted_output, image) # --> M X (x, y, x, y, id, conf, cls, ind)
    tracker.plot_results(image, show_trajectories=True)

video.release()

    # # break on pressing q or space
    # cv2.imshow('BoxMOT detection', image)     
    # key = cv2.waitKey(1) & 0xFF
    # if key == ord(' ') or key == ord('q'):
    #     break

# cv2.destroyAllWindows()
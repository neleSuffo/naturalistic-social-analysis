import cv2
import sys
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Add leuphana_ipe directory to the system path
sys.path.append('/home/nele_pauline_suffo/projects/boxmot')
from boxmot import DeepOCSORT

yolo_model = YOLO('/home/nele_pauline_suffo/models/yolov8_trained.pt')

tracker = DeepOCSORT(
    model_weights=Path('/home/nele_pauline_suffo/models/duke_R101.onnx'), # which ReID model to use
    device='cuda:0',
    fp16=False,
)

# Define the path to your video file
video_path = '/home/nele_pauline_suffo/ProcessedData/videos/quantex_at_home_id254922_2022_04_12_01.MP4'

# Open the video file
vid = cv2.VideoCapture(video_path)

while True:
    model = yolo_model.model
    ret, im = vid.read()

    # substitute by your object detector, output has to be N X (x, y, x, y, conf, cls)
    dets = model(im, size=640)  # --> N X (x, y, x, y, conf, cls)

    # Check if there are any detections
    if dets.size > 0:
        tracker.update(dets, im) # --> M X (x, y, x, y, id, conf, cls, ind)
    # If no detections, make prediction ahead
    else:   
        dets = np.empty((0, 6))  # empty N X (x, y, x, y, conf, cls)
        tracker.update(dets, im) # --> M X (x, y, x, y, id, conf, cls, ind)
    tracker.plot_results(im, show_trajectories=True)

    # break on pressing q or space
    cv2.imshow('BoxMOT detection', im)     
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') or key == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
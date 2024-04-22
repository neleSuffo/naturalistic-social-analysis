from ultralytics import YOLO
#import os
#from IPython.display import display, Image
#from IPython import display
#display.clear_output
#!yolo checks

data_path = 'datasets/coco128/coco128.yaml'

# Load a model
model = YOLO("yolov8x.pt") # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
#results = model.train(data=data_path, epochs=20, imgsz=640, device='mps')
results = model.train(data=data_path, epochs=3, imgsz=640)

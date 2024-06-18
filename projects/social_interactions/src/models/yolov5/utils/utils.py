# Load YOLOv5 model
import logging
import torch

def load_yolo_model():
    """ 
    Load the YOLOv5 model for person detection.
    """
    try:
        yolo_model = torch.hub.load("ultralytics/yolov5", "yolov5s")
        logging.info("Successfully downloaded and loaded the yolov5 model.")
        return yolo_model
    except Exception as e:
        logging.error(f"Error occurred while loading the yolov5 model: {e}")
        raise
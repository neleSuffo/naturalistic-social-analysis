from constants import DetectionParameters
from facenet_pytorch import MTCNN
from fast_mtcnn import FastMTCNN
import os
import json
import shutil
import torch
import logging


# TODO: Add proximity detection
def run_proximity_detection():
    pass  # Implement proximity detection here


def load_person_detection_model():
    """
    This function loads the person detection model.

    Returns
    -------
    torch.nn.Module
        the person detection model
    """
    # Load the YOLOv5 model for person detection
    person_detection_model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    # Move the model to the desired location
    if os.path.exists("yolov5s.pt"):
        shutil.move(
            "yolov5s.pt",
            "/Users/nelesuffo/projects/leuphana-IPE/pretrained_models/yolov5s.pt",
        )

    return person_detection_model


def load_batch_face_detection_model():
    """
    This function loads the face detection model.

    Returns
    -------
    MTCNN
        the face detection model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the batch MTCNN model
    face_detection_model = FastMTCNN(
        stride=2, resize=1, margin=14, factor=0.6, keep_all=True, device=device
    )
    return face_detection_model


def load_frame_face_detection_model():
    """
    This function loads the face detection model.

    Returns
    -------
    MTCNN
        the face detection model
    """
    # Load the full MTCNN model
    return MTCNN()


def save_results_to_json(results: dict, output_path: str) -> None:
    """
    This function saves the results to a JSON file.

    Parameters
    ----------
    results : dict
        the results for each detection type
    output_path : str
        the path to the output file
    """
    directory = os.path.dirname(output_path)
    # Check if the output directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Save the results to a JSON file
    with open(output_path, "w") as f:
        json.dump(results, f)


def display_results(results: dict) -> None:
    """
    This function calculates the percentage of frames where the object is detected
    and prints the results.

    Parameters
    ----------
    results : dict
        the results for each detection type
    """
    for detection_type, detection_list in results.items():
        percentage = sum(detection_list) / len(detection_list) * 100
        logging.info(
            f"Percentages of at least one {detection_type} detected relative to the total number of frames: {percentage:.2f}"
        )
        logging.info(
            f"Total number of steps (every {DetectionParameters.frame_step}-th frame) ({detection_type}): {len(detection_list)}"
        )

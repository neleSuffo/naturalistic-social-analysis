import os
import shutil
import torch
from facenet_pytorch import MTCNN


def load_detection_models(person_detection: bool, face_detection: bool) -> tuple:
    """
    This function loads the person detection and face detection models.

    Parameters
    ----------
    person_detection : bool
        whether to load the person detection model
    face_detection : bool
        whether to load the face detection model

    Returns
    -------
    tuple
        the person detection model and the face detection model
    """
    person_detection_model = None
    face_detection_model = None

    if person_detection:
        # Load the YOLOv5 model for person detection
        person_detection_model = torch.hub.load("ultralytics/yolov5", "yolov5s")
        # Move the model to the desired location
        if os.path.exists("yolov5s.pt"):
            shutil.move(
                "yolov5s.pt",
                "/Users/nelesuffo/projects/leuphana-IPE/pretrained_models/yolov5s.pt",
            )

    if face_detection:
        # Load the MTCNN model for face detection
        face_detection_model = MTCNN()

    return person_detection_model, face_detection_model


def calculate_percentage_and_print_results(
    detection_list: list, detection_type: str
) -> None:
    """
    This function calculates the percentage of frames
    where the object is detected and prints the results.

    Parameters
    ----------
    detection_list : list
        the list of detections
    detection_type : str
        the type of detection
    """
    percentage = sum(detection_list) / len(detection_list) * 100
    print(
        f"Percentages of at least one {detection_type} detected relative to the total frames: {percentage:.2f}"  # noqa: E231, E501
    )
    print(f"Total number of frames ({detection_type}): {len(detection_list)}")

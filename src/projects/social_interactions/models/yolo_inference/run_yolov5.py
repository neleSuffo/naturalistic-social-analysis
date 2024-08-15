from src.projects.social_interactions.common.constants import DetectionPaths, DetectionParameters
from src.projects.social_interactions.common.my_utils import create_video_writer
from src.projects.social_interactions.config.config import generate_detection_output_video
from typing import Optional
import numpy as np
import cv2
import os
import torch
import json
import multiprocessing


def run_yolo(
    video_input_path: str,
    annotation_id: multiprocessing.Value,
    image_id: multiprocessing.Value,
    video_file_name: str,
    file_id: int,
    model: torch.nn.Module,
    existing_image_file_names: list,
) -> Optional[dict]:
    """
    This function performs person detection on a video file (every frame_step-th frame)
    The detection results are returned in COCO format or saved to a video file.

    Parameters
    ----------
    video_input_path : str
        the path to the video file
    annotation_id : multiprocessing.Value
        the unique annotation ID
    image_id : multiprocessing.Value
        the unique image ID
    video_file_name : str
        the name of the video file
    file_id: int
        the unique video file id
    model : torch.nn.Module
        the YOLOv5 model
    existing_image_file_names : list
        the image file names that already exist in the combined json output file


    Returns
    -------
    dict
        the detection results in COCO format
    """
    # Check if the video file exists and is accessible
    if not video_input_path.exists():
        raise FileNotFoundError(
            f"The video file {video_input_path} does not exist or is not accessible."
        )

    # Check if the model is a valid YOLO model
    if not isinstance(model, torch.nn.Module):
        raise ValueError("The model is not a valid YOLO model.")

    # Get the class labels and class index
    class_list = model.names
    class_index_det = [
        key
        for key, value in class_list.items()
        if value == DetectionParameters.yolo_detection_class
    ][0]

    # Load video file
    cap = cv2.VideoCapture(video_input_path)

    # If a detection output video should be generated
    if generate_detection_output_video:
        # Load video file and extract properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = int(cap.get(cv2.CAP_PROP_FPS))

        # Create video output directory if it does not exist
        video_output_path = DetectionPaths.person / video_file_name

        # Create a VideoWriter object to write the output video
        out = create_video_writer(
            video_output_path, frames_per_second, frame_width, frame_height
        )
        # Perform detection and write output video
        detection_video_output(
            cap,
            out,
            video_input_path,
            video_output_path,
            model,
            person_detection_with_bbox,
            person_draw_fn,
            class_index_det,
        )
    else:
        # Perform detection and return results in
        detection_output = detection_coco_output(
            cap,
            model,
            person_detection_without_bbox,
            person_process_results,
            video_file_name,
            file_id,
            annotation_id,
            image_id,
            existing_image_file_names,
            class_index_det,
        )

        return detection_output


def person_detection_without_bbox(
    model: torch.nn.Module,
    frame: np.ndarray,
) -> tuple:
    """
    This function performs person detection on a frame using the YOLOv5 model.

    Parameters
    ----------
    model : torch.nn.Module
        the YOLOv5 model
    frame : np.ndarray
        the frame to process

    Returns
    -------
    tuple
        the detection results
    """
    return model(frame)


def person_process_results(
    results: dict,
    detection_output: dict,
    category_id: int,
    annotation_id: multiprocessing.Value,
    image_id: multiprocessing.Value,
    class_index_det: int,
):
    """
    This function processes the detection results for person detection.

    Parameters
    ----------
    results : dict
        the detection results
    detection_output : dict
        the detection output in COCO format
    category_id : int
        the category ID for person detection
    annotation_id : multiprocessing.Value
        the unique annotation ID
    image_id : multiprocessing.Value
        the unique image ID
    class_index_det : int
        the class index of the class to detect in the yolo model
    """
    for result in results.pred:
        for det in result:
            x1, y1, x2, y2, conf, cls = det
            # Check if the detected class is equal
            # to the class index of interest
            if int(cls) == class_index_det:
                detection_output["annotations"].append(
                    {
                        "id": annotation_id.value,
                        "image_id": image_id.value,
                        "category_id": category_id,
                        # Convert bbox values to list
                        "bbox": [
                            x1.item(),
                            y1.item(),
                            (x2 - x1).item(),
                            (y2 - y1).item(),
                        ],
                        # Convert score to float
                        "score": conf.item(),
                    }
                )
                with annotation_id.get_lock():
                    annotation_id.value += 1


def person_detection_with_bbox(
    model: torch.nn.Module,
    frame: torch.Tensor,
    class_index_det: int,
) -> list:
    """This function performs person detection on a frame using the YOLOv5 model.

    Parameters
    ----------
    model : torch.nn.Module
        the YOLOv5 model
    frame : torch.Tensor
        the frame to process
    class_index_det : int
        the class index of the class to detect for the yolo model

    Returns
    -------
    list
        the detection results
    """
    # Perform detection
    results = model(frame)
    # Initialize list to store detections
    detections = []
    for result in results.pred:
        for det in result:
            x1, y1, x2, y2, conf, cls = det
            # Check if the detected class is equal
            # to the class index of interest
            if int(cls) == class_index_det:
                detections.append((x1, y1, x2, y2, conf, cls))
    return detections


def person_draw_fn(
    frame: torch.Tensor,
    detection: tuple,
) -> None:
    """This function draws the detection results on the frame.

    Parameters
    ----------
    frame : torch.Tensor
        the frame to draw the detection results on
    detection : tuple
        the detection results
    """
    x1, y1, x2, y2, conf, cls = detection
    cv2.rectangle(
        frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (146, 123, 45),
        2,
    )
    cv2.putText(
        frame,
        f"person {conf:.2f}",
        (int(x1), int(y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (146, 123, 45),
        2,
    )

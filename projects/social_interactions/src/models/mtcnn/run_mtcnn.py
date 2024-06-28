from facenet_pytorch import MTCNN
from projects.social_interactions.src.common.constants import DetectionPaths
from projects.social_interactions.src.common.my_utils import (
    create_video_writer,
    detection_coco_output,
    detection_video_output,
)
from projects.social_interactions.src.config.config import generate_detection_output_video
from typing import Optional
import numpy as np
import cv2
import os
import multiprocessing


def run_face_detection(
    video_input_path: str,
    annotation_id: multiprocessing.Value,
    video_file_name: str,
    file_id: str,
    model: MTCNN,
) -> Optional[dict]:
    """
    This function performs frame-wise face detection on a video file (every frame_step-th frame)
    If generate_detection_output_video is True, it creates a VideoWriter object to write the output video.
    Otherwise it returns the detections in COCO format.

    Parameters
    ----------
    video_input_path : str
        the path to the video file
    annotation_id : multiprocessing.Value
        the annotation ID
    video_file_name : str
        the name of the video file
    file_id: str,
        the video file id
    model : MTCNN
        the MTCNN face detector


    Returns
    -------
    dict
        the detection results in COCO format
    """
    # Check if the video file exists and is accessible
    if not os.path.isfile(video_input_path):
        raise FileNotFoundError(
            f"The video file {video_input_path} does not exist or is not accessible."
        )

    # Check if the model is a valid MTCNN model
    if not isinstance(model, MTCNN):
        raise ValueError("The model is not a valid MTCNN model.")

    # Load video file
    cap = cv2.VideoCapture(video_input_path)

    # If a detection output video should be generated
    if generate_detection_output_video:
        # Extract video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = int(cap.get(cv2.CAP_PROP_FPS))

        # Create video output directory if it does not exist
        video_output_path = os.path.join(DetectionPaths.face, video_file_name)

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
            face_detection_fn_long,
            draw_faces,
        )
    else:
        # Perform detection and return results in COCO format
        detection_output = detection_coco_output(
            cap,
            model,
            face_detection_without_bbox,
            face_process_results,
            video_file_name,
            file_id,
            annotation_id,
        )

        return detection_output


def face_detection_without_bbox(
    model: MTCNN,
    frame: np.ndarray,
) -> tuple:
    """
    This function performs face detection on a frame using the MTCNN model.

    Parameters
    ----------
    model : MTCNN
        the MTCNN model
    frame : np.ndarray
        the frame to process

    Returns
    -------
    tuple
        the detection results
    """
    return model.detect(frame)


def face_process_results(
    boxes_and_probs: tuple,
    detection_output: dict,
    frame_count: int,
    category_id: int,
    annotation_id: multiprocessing.Value,
    class_index_det: int,
):
    """
    This function processes the detection results for face detection.

    Parameters
    ----------
    boxes_and_probs : tuple
        the boxes and probabilities
    detection_output : dict
        the detection output in COCO format
    frame_count : int
        the frame count
    category_id : int
        the category ID for face detection
    annotation_id : multiprocessing.Value
        the annotation ID
    class_index_det : int
        a placeholder argument
    """
    boxes, probs = boxes_and_probs
    if boxes is not None:
        for box, prob in zip(boxes, probs):
            x1, y1, x2, y2 = box
            # Add annotation information to COCO output
            detection_output["annotations"].append(
                {
                    "id": annotation_id.value,
                    "image_id": frame_count,
                    "category_id": category_id,
                    "bbox": [x1, y1, (x2 - x1), (y2 - y1)],
                    "score": prob,
                }
            )
            with annotation_id.get_lock():
                annotation_id.value += 1


def face_detection_with_bbox(
    model: MTCNN,
    frame: np.ndarray,
    class_index_det: int,
) -> tuple:
    """
    This function performs face detection on a frame using the MTCNN model.

    Parameters
    ----------
    model : MTCNN
        the MTCNN model
    frame : np.ndarray
        the frame to process

    Returns
    -------
    tuple
        the detection results
    """
    boxes, probs = model.detect(frame)
    detections = []
    if boxes is not None:
        for box, prob in zip(boxes, probs):
            x1, y1, x2, y2 = box
            detections.append((x1, y1, x2, y2, prob))
    return detections


def draw_faces(
    frame: np.ndarray,
    detection: tuple,
) -> None:
    """
    This function draws the face detection results on the frame.

    Parameters
    ----------
    frame : np.ndarray
        the frame to draw on
    detection : tuple
        the detection results
    """
    x1, y1, x2, y2, prob = detection
    cv2.rectangle(
        frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (105, 22, 51),
        2,
    )
    cv2.putText(
        frame,
        f"face {prob:.2f}",
        (int(x1), int(y1) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (105, 22, 51),
        2,
    )

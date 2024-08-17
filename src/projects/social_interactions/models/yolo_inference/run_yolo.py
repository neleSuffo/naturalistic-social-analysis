import cv2
import os
import torch
import json
import multiprocessing
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional
from src.projects.social_interactions.common.constants import DetectionPaths, DetectionParameters as DP, LabelToCategoryMapping
from src.projects.social_interactions.common.my_utils import create_video_writer
from src.projects.social_interactions.config.config import generate_detection_output_video


def run_yolo(
    video_file: Path,
    video_file_ids_dict: dict,
    annotation_id: multiprocessing.Value,
    image_id: multiprocessing.Value,
    model: torch.nn.Module,
    existing_image_file_names: list,
) -> Optional[dict]:
    """
    This function performs frame-wise person detection on a video file using a YOLO model.
    If generate_detection_output_video is True, it creates a VideoWriter object to write the output video.
    Otherwise, it returns the detections in COCO format.

    Parameters
    ----------
    video_file : Path
        The path to the video file.
    video_file_ids_dict : dict
        Dictionary mapping video filenames to their corresponding IDs.
    annotation_id : multiprocessing.Value
        The shared annotation ID for COCO format.
    image_id : multiprocessing.Value
        The shared image ID for COCO format.
    model : torch.nn.Module
        The YOLOv5 model.
    existing_image_file_names : list
        List of image file names already existing in the combined JSON output.

    Returns
    -------
    dict
        The detection results in COCO format if no output video is generated, otherwise None.
    """
    # Validate inputs
    validate_inputs(video_file, model)
    
    video_file_name = video_file.stem
    file_id = video_file_ids_dict.get(video_file_name)

    # Load video file
    cap = cv2.VideoCapture(str(video_file))
    
    # If a detection output video should be generated
    if generate_detection_output_video:
        video_output_path = DetectionPaths.person / f"{video_file_name}.mp4"
        process_and_save_video(
            cap, 
            video_file,
            video_output_path, 
            model)
        return None
    else:
        return detection_coco_output(
            cap, 
            model, 
            video_file_name, 
            file_id, 
            annotation_id, 
            image_id, 
            existing_image_file_names
        )

def validate_inputs(video_file: Path, model: torch.nn.Module):
    """
    Validates the video input path and YOLO model.

    Parameters
    ----------
    video_file : Path
        Path to the input video file.
    model : torch.nn.Module
        The YOLO model instance.

    Raises
    ------
    FileNotFoundError
        If the video file does not exist or is not accessible.
    ValueError
        If the provided model is not an instance of torch.nn.Module.
    """
    if not video_file.exists():
        raise FileNotFoundError(f"The video file {video_file} does not exist or is not accessible.")

    if not isinstance(model, torch.nn.Module):
        raise ValueError("The model is not a valid YOLO model.")


def process_and_save_video(
    cap: cv2.VideoCapture, 
    video_output_path: Path, 
    model: torch.nn.Module):
    """
    Processes the video frame-by-frame, applies person detection, and saves the output video.

    Parameters
    ----------
    cap : cv2.VideoCapture
        The video capture object.
    video_output_path : Path
        The path where the output video should be saved.
    model : torch.nn.Module
        The YOLO model instance.
    """
    # Extract video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = int(cap.get(cv2.CAP_PROP_FPS))

    # Create a VideoWriter object to write the output video
    out = create_video_writer(
        video_output_path, frames_per_second, frame_width, frame_height
    )
    
    # Perform detection and write output video
    detection_video_output(
        cap,
        out,
        model,
    )

def detection_coco_output(
    cap: cv2.VideoCapture,
    model: torch.nn.Module,
    video_file_name: str,
    file_id: int,
    annotation_id: multiprocessing.Value,
    image_id: multiprocessing.Value,
    existing_image_file_names: list,
) -> dict:
    """
    This function performs detection on a video file and returns the detection results in COCO format.

    Parameters
    ----------
    cap : cv2.VideoCapture
        The video capture object.
    model : torch.nn.Module
        The YOLO model used for person detection.
    video_file_name : str
        The name of the video file.
    file_id : int
        The video file ID.
    annotation_id : multiprocessing.Value
        The unique annotation ID.
    image_id : multiprocessing.Value
        The unique image ID.
    existing_image_file_names : list
        The list of image file names already in the detection output JSON.

    Returns
    -------
    dict
        The detection results in COCO format.
    """
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Initialize the detection output
    detection_output = {
        DP.output_key_images: [],
        DP.output_key_annotations: [],
        DP.output_key_categories: [],
    }
    
    # Determine the category ID for person detection
    category_id = LabelToCategoryMapping.label_dict[
        DP.yolo_detection_class
    ]
    
    # Get the class index for person detection for the YOLO model
    class_index_det = next(
        key for key, value in model.names.items()
        if value == DP.yolo_detection_class
    )
    
    # Process each frame
    with tqdm(total=total_frames, desc="Processing frames", ncols=70) as pbar:
        for frame_count in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            pbar.update()
            
            if frame_count % DP.frame_step == 0:
                results = model(frame)

                if results.pred[0] is not None:
                    file_name = f"{video_file_name}_{frame_count:06}.jpg"
                    # Only add the image file name if it does not already exist
                    if file_name not in existing_image_file_names:
                        detection_output[DP.output_key_images].append({
                            "id": image_id.value,
                            "video_id": file_id,
                            "frame_id": frame_count,
                            "file_name": file_name,
                        })
                        existing_image_file_names.append(file_name)
                        
                        # Add the annotations for the detected persons
                        for det in results.pred[0]:
                            x1, y1, x2, y2, conf, cls = det
                            if int(cls) == class_index_det:
                                detection_output[DP.output_key_annotations].append({
                                    "id": annotation_id.value,
                                    "image_id": image_id.value,
                                    "category_id": category_id,
                                    "bbox": [x1.item(), y1.item(), (x2 - x1).item(), (y2 - y1).item()],
                                    "score": conf.item(),
                                })
                                with annotation_id.get_lock():
                                    annotation_id.value += 1
                                    
                        # Add the person category if it does not exist
                        if category_id not in [category['id'] for category in detection_output[DP.output_key_categories]]:
                            detection_output[DP.output_key_categories].append({
                                "id": category_id,
                                "name": DP.yolo_detection_class
                            })

                        with image_id.get_lock():
                            image_id.value += 1

    return detection_output

def detection_video_output(
    cap: cv2.VideoCapture,
    out: cv2.VideoWriter,
    model: torch.nn.Module,
) -> None:
    """
    This function performs frame-wise person detection on a video file and writes the output video with bounding boxes.

    Parameters
    ----------
    cap : cv2.VideoCapture
        The video capture object.
    out : cv2.VideoWriter
        The video writer object.
    model : torch.nn.Module
        The YOLO model used for person detection.
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    class_index_det = next(
        key for key, value in model.names.items()
        if value == DP.yolo_detection_class
    )
    
    # Process each frame
    with tqdm(total=total_frames, desc="Processing frames", ncols=70) as pbar:
        for frame_count in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            pbar.update()

            if frame_count % DP.frame_step == 0:
                results = model(frame)

                if results.pred[0] is not None:
                    for det in results.pred[0]:
                        x1, y1, x2, y2, conf, cls = det
                        if int(cls) == class_index_det:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (146, 123, 45), 2)
                            cv2.putText(frame, f"person {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (146, 123, 45), 2)

            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
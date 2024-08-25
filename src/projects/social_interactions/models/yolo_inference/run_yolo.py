import cv2
import torch
import logging
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
from typing import Optional
from src.projects.social_interactions.common.constants import DetectionPaths, DetectionParameters as DP, LabelToCategoryMapping
from src.projects.social_interactions.common.my_utils import create_video_writer
from src.projects.social_interactions.config.config import generate_detection_output_video


def run_yolo(
    video_file: Path,
    model: torch.nn.Module,
) -> Optional[dict]:
    """
    This function performs frame-wise person detection on a video file using a YOLO model.
    If generate_detection_output_video is True, it creates a VideoWriter object to write the output video.
    Otherwise, it returns the detections in COCO format.

    Parameters
    ----------
    video_file : Path
        The path to the video file.
    model : torch.nn.Module
        The YOLOv5 model.

    Returns
    -------
    dict
        The detection results in COCO format if no output video is generated, otherwise None.
    """
    # Validate inputs
    validate_inputs(video_file, model)
    
    video_file_name = video_file.stem

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
        logging.info(f"Detection output video generated at {video_output_path}.")
        return None
    else:
        detection_results = detection_json_output(
            cap, 
            model, 
            video_file_name, 
        )
        logging.info("Detection results generated.")
        return detection_results


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

def detection_json_output(
    cap: cv2.VideoCapture,
    model: YOLO,
    video_file_name: str,
) -> dict:
    """
    This function performs detection on a video file 
    It returns the detection results in JSON format.

    Parameters
    ----------
    cap : cv2.VideoCapture
        The video capture object.
    model : YOLO
        The YOLO model used for person detection.
    video_file_name : str
        The name of the video file.

    Returns
    -------
    list
        The detection results in JSON format.
    """
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize the detection output for the video
    detection_output = {
        DP.output_key_images: [],
        DP.output_key_categories: []
    }    
    
    # Process each frame
    with tqdm(total=total_frames, desc="Processing frames", ncols=70) as pbar:
        for frame_count in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Frame {frame_count} could not be read. Skipping.")
                break
            
            # Update progress bar
            pbar.update()
            
            if frame_count % DP.frame_step == 0:
                # Perform detection on the current frame
                results = model(frame)
                
                # Create the file name for the current image
                file_name = f"{video_file_name}_{frame_count:06}.jpg"
                
                # Initialize the dictionary for the current image
                image_detections = {
                    "image_id": file_name,
                    DP.output_key_annotations: []
                }
                
                # Add the annotations for the detected objects
                for boxes in results[0].boxes:
                    # Get the bounding box coordinates (x center, y center, width, height) normalized to the image size
                    x_center, y_center, width, height = boxes.xywhn[0]
                    conf = boxes.conf[0]
                    cls = boxes.cls[0]
                    # Get the category ID and name
                    category_id = int(cls.item())
                    category_name = model.names[category_id]
                    
                    # Append the detection data to the list for this image
                    image_detections[DP.output_key_annotations].append({
                        "class": category_name,
                        "bbox": [x_center.item(), y_center.item(), width.item(), height.item()],
                        "confidence": conf.item()
                    })
                # Add the image's detection data to the video-level output
                detection_output[DP.output_key_images].append(image_detections)      

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
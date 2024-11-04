from facenet_pytorch import MTCNN
from pathlib import Path
from moviepy.editor import VideoFileClip
from constants import DetectionPaths
from utils import create_video_writer
from config import generate_detection_output_video, DetectionParameters as DP
from typing import Optional
from tqdm import tqdm
from PIL import Image
import cv2
import logging
import numpy as np

def run_mtcnn(
    video_file: Path,
    model: MTCNN,
) -> Optional[dict]:
    """
    This function performs frame-wise face detection on a video file (every frame_step-th frame)
    If generate_detection_output_video is True, it creates a VideoWriter object to write the output video.
    Otherwise it returns the detections in COCO format.
    
    Parameters
    ----------
    video_file : Path
        the path to the video file.
    model : MTCNN
        the MTCNN face detector.
    
    Returns
    -------
    dict
        the detection results in COCO format if no output video is generated, otherwise None.
    """
    # Validate inputs
    validate_inputs(video_file, model)
    
    video_file_name = video_file.stem

    # Load video file
    cap = cv2.VideoCapture(str(video_file))
    
    # If a detection output video should be generated
    if generate_detection_output_video:
        video_output_path = DetectionPaths.face_detections_dir / f"{video_file_name}.mp4"
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


def validate_inputs(video_input_path: Path, model: MTCNN):
    """
    Validates the video input path and MTCNN model.

    Parameters
    ----------
    video_input_path : Path
        Path to the input video file.
    model : MTCNN
        The MTCNN model instance.

    Raises
    ------
    FileNotFoundError
        If the video file does not exist or is not accessible.
    ValueError
        If the provided model is not an instance of MTCNN.
    """
    # Check if the video file exists and is accessible
    if not video_input_path.exists():
        raise FileNotFoundError(f"The video file {video_input_path} does not exist or is not accessible.")

    # Check if the model is a valid MTCNN model
    if not isinstance(model, MTCNN):
        raise ValueError("The model is not a valid MTCNN model.")


def process_and_save_video(
    cap: cv2.VideoCapture, 
    video_file: Path,
    video_output_path: Path, 
    model: MTCNN):
    """
    Processes the video frame-by-frame, applies face detection, and saves the output video.

    Parameters
    ----------
    cap : cv2.VideoCapture
        The video capture object.
    video_file : Path
        The path to the input video file.
    video_output_path : Path
        The path where the output video should be saved.
    model : MTCNN
        The MTCNN model instance.
    """
    # Extract video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = int(cap.get(cv2.CAP_PROP_FPS))

    # Create a VideoWriter object to write the output video
    out = create_video_writer(
        video_output_path, frames_per_second, frame_width, frame_height
    )

    # Perform detection and write the output video
    detection_video_output(
        cap,
        out,
        str(video_file),
        str(video_output_path),
        model,
    )


def detection_json_output(
    cap: cv2.VideoCapture,
    model: MTCNN,
    video_file_name: str,
) -> dict:
    """
    This function performs detection on a video file 
    It returns the detection results in JSON format.
    
    Parameters
    ----------
    cap : cv2.VideoCapture
        The video capture object.
    model : MTCNN
        The MTCNN model used for face detection.
    video_file_name : str
        The name of the video file.

    Returns
    -------
    dict
        The detection results in JSON format.
    """
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Initialize detection output
    detection_output = {
        DP.output_key_images: [], 
        DP.output_key_categories: [],
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
            
            # Perform detection every frame_step-th frame
            if frame_count % DP.frame_step_interval == 0:
                # Convert the image from BGR (OpenCV default) to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert the NumPy array to a PIL Image
                frame = Image.fromarray(frame)
                # Resizing 
                #TODO: Check if resizing is necessary
                frame = frame.resize([int(f * 0.25) for f in frame.size])
                
                # Perform face detection using MTCNN
                boxes, probs = model.detect(frame)
                
                # Create the file name for the current image
                file_name = f"{video_file_name}_{frame_count:06}.jpg"
                
                # Initialize the dictionary for the current image
                image_detections = {
                    "image_id": file_name,
                    DP.output_key_annotations: []
                }
                
                # Process detection results and add annotations to detection output
                if boxes is not None:
                    # Process detection results and add annotations to detection output
                    for box, prob in zip(boxes, probs):
                        x1, y1, x2, y2 = box
                        image_detections[DP.output_key_annotations].append({
                            "class": DP.mtcnn_detection_target,
                            "bbox": [x1, y1, x2, y2],
                            "confidence": prob,
                        })
                # Add the image's detection data to the video-level output
                detection_output[DP.output_key_images].append(image_detections)      

    return detection_output


def detection_video_output(
    cap: cv2.VideoCapture,
    out: cv2.VideoWriter,
    video_input_path: str,
    video_output_path: str,
    model: MTCNN,
) -> None:
    """
    This function performs frame-wise face detection on a video file and writes the output video with bounding boxes.

    Parameters
    ----------
    cap : cv2.VideoCapture
        The video capture object.
    out : cv2.VideoWriter
        The video writer object.
    video_input_path : str
        The path to the input video file.
    video_output_path : str
        The path to the output video file.
    model : MTCNN
        The MTCNN model used for face detection.
    """
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a progress bar
    with tqdm(total=total_frames, desc="Processing frames", ncols=70) as pbar:
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            pbar.update()

            # Perform face detection every frame_step-th frame
            if frame_count % DP.frame_step == 0:
                # Perform face detection on the current frame
                boxes, probs = model.detect(frame)

                # If faces are detected, draw bounding boxes and probabilities
                if boxes is not None:
                    for box, prob in zip(boxes, probs):
                        x1, y1, x2, y2 = box
                        # Draw bounding box and probability on the frame
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (105, 22, 51), 2)
                        cv2.putText(frame, f"face {prob:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (105, 22, 51), 2)

            # Write the processed frame to the output video
            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Add audio to the output video
        clip = VideoFileClip(video_output_path)
        processed_filename = f"{video_output_path.stem}_processed{video_output_path.suffix}"
        clip.write_videofile(processed_filename, codec="libx264", audio=video_input_path)

        # Delete the video file without audio
        video_output_path.unlink()
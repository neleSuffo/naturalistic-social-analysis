import cv2
import sys
import torch
import pickle
import numpy as np
import logging
from pathlib import Path
from itertools import chain
from typing import Tuple, Dict
from src.constants import YoloPaths, StrongSortPaths, DetectionPaths
from src.config import DetectionParameters
from ultralytics import YOLO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add boxmot to the Python path
sys.path.append('/home/nele_pauline_suffo/projects/boxmot')
from boxmot import DeepOcSort, StrongSort

# Initialize YOLO model and tracker
yolo_model = YOLO('/home/nele_pauline_suffo/models/yolov8_trained.pt')

deep_oc_sort_tracker = DeepOcSort(
    reid_weights=Path('resnet50_dukemtmcreid.pt'),  # which ReID model to use
    device='cuda:0',
    half=False,
)

strongsort_tracker = StrongSort(
    reid_weights=Path('resnet50_dukemtmcreid.pt'),  # which ReID model to use
    device='cuda:0',
    half=False,
)   
    
def extract_detections(video_path: Path, output_folder: Path):
    """
    This function extracts detections from a video file and saves them to a .npy file.
    
    Parameters
    ----------
    video_path : Path
        the path to the video file
    output_folder : Path
        the folder where the detections will be saved
    """
    video_name = video_path.stem
    detections_path = output_folder / f"{video_name}_detections.npy"
    
    # Initialize the video capture
    video = cv2.VideoCapture(str(video_path))
    frame_index = 0
    
    all_detections = {}
        
    logging.info(f"Processing video: {video_name}")

    while True:
        ret, image = video.read()
        
        if not ret:
            logging.info(f"End of video reached: {video_name}")
            break
        
        detections = run_yolo_model_on_image(image)

        all_detections[frame_index] = detections
        frame_index += 1

    # Save all detections to a file using pickle
    with open(detections_path, 'wb') as f:
        pickle.dump(all_detections, f)
    logging.info(f"Saved detections for {video_name} to {detections_path}")

    # Release resources
    video.release()   


def run_yolo_model_on_image(image: np.ndarray) -> np.ndarray:
    """
    This function runs the YOLO model on the input image and returns the detection results.

    Parameters
    ----------
    image : np.ndarray
        the input image to run the YOLO model on

    Returns
    -------
    np.ndarray
        the detection results in the format N x (x, y, x, y, conf, cls)
    """
    # Run the YOLO model on the image
    logging.info("Running YOLO model on image")
    results = yolo_model(image)
    # Convert the detections to the format expected by the tracking model
    detection = results[0].boxes
    cls = detection.cls.cpu().numpy() if torch.is_tensor(detection.cls) else detection.cls
    conf = detection.conf.cpu().numpy() if torch.is_tensor(detection.conf) else detection.conf
    xyxy = detection.xyxy.cpu().numpy() if torch.is_tensor(detection.xyxy) else detection.xyxy
    output = np.column_stack((xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3], conf, cls))
    return output


def run_tracking_on_detections(
    detections_path: Path, 
    output_folder: Path, 
    tracker: str,
    save_video: bool = True):
    """
    This function runs the tracking model on the given detections and saves the tracking results.

    Parameters
    ----------
    detections_path : Path
        the path to the detections file
    output_folder : Path
        the folder where the tracking results will be saved
    """
    video_name = detections_path.stem.replace('_detections', '')
    # Extract the stem and remove the "_detections" suffix
    output_path = output_folder / f"{video_name}_{tracker}.npy"
    video_path = get_video_path(video_name, DetectionPaths.videos_input_dir)
    
    all_detections = np.load(detections_path, allow_pickle=True)
    
    logging.info(f"Loaded detections from {detections_path}")

    # Open video file for reading frames
    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        return

    # # Initialize the tracking data
    # all_tracking_data = {
    #     'tracked_boxes': {},
    #     'image_names': [],
    #     'images': []
    # }
    if save_video:
        # Prepare the output video writer
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)

        output_video_path = output_folder / f"{video_name}_{tracker}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))
        
    frame_index = 0
    while True:
        # Load the next frame directly from video
        ret, image = video.read()
        if not ret:
            logging.info(f"End of video reached: {video_name}")
            break
        
        image_name = f"{video_name}_{frame_index:06d}.png"

        # Check if there's a corresponding detection in the dictionary
        detections = all_detections.get(frame_index, np.empty((0, 6)))         

        # Run the tracking model on the detections      
        if tracker == 'deep_oc_sort':
            tracked_image = run_deep_oc_sort_model(detections, image)
        elif tracker == 'strong_sort':
            tracked_image = run_strong_sort_model(detections, image)
        else:
            logging.error(f"Unknown tracker: {tracker}. Please use 'deep_oc_sort' or 'strong_sort'.")
            return      

        # Run the tracking model on the detections
        if save_video:
            # Write the processed frame to the output video
            video_writer.write(tracked_image)
        
        # all_tracking_data['tracked_boxes'].update(tracking_data['tracked_boxes'])
        # all_tracking_data['images'].append(tracked_image)
        # all_tracking_data['image_names'].append(image_name)
        
        frame_index += 1

    #np.save(output_path, all_tracking_data)
    #logging.info(f"Saved tracking results for {video_name} to {output_path}")
    
    if save_video:
        video.release()
        video_writer.release()
        logging.info(f"Tracking results saved to {output_video_path}") 

    
def get_video_path(video_name: str, videos_dir: Path) -> Path:
    """
    This function searches for a video file with case-insensitive extensions (.mp4, .MP4).

    Parameters
    ----------
    video_name : str
        The base name of the video without the extension.
    videos_dir : Path
        The directory where video files are stored.

    Returns
    -------
    Path
        The path to the video file if found, otherwise raises a FileNotFoundError.
    """
    # Search for files with both .mp4 and .MP4 extensions (case insensitive)
    possible_videos = list(videos_dir.glob(f"{video_name}.*"))  # Matches any extension
    for video in possible_videos:
        if video.suffix.lower() == DetectionParameters.video_file_extension:
            return video
    
    raise FileNotFoundError(f"Video file for {video_name} not found in {videos_dir}")


def run_deep_oc_sort_model(detections: np.ndarray, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    This function runs the DeepSORT tracker on the given detections and image.

    Parameters
    ----------
    detections : np.ndarray
        the detection results in the format N x (x, y, x, y, conf, cls)
    image : np.ndarray
        the input image to run the tracker on

    Returns
    -------
    np.ndarray
        the image with the tracking results
    dict
        a dictionary containing the tracking data
    """
    # Check if there are any detections
    if detections.size > 0:
        deep_oc_sort_tracker.update(detections, image)  # M X (x, y, x, y, id, conf, cls, ind)
    else:
        detections = np.empty((0, 6))  # empty N X (x, y, x, y, conf, cls)
        deep_oc_sort_tracker.update(detections, image)  # M X (x, y, x, y, id, conf, cls, ind)
    
    # Get tracking results and plot them on the image
    tracked_image = deep_oc_sort_tracker.plot_results(image, show_trajectories=True)
    
    return tracked_image


def run_strong_sort_model(detections: np.ndarray, image: np.ndarray, embeddings: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
    """
    This function runs the StrongSORT tracker on the given detections and image.

    Parameters
    ----------
    detections : np.ndarray
        the detection results in the format N x (x, y, x, y, conf, cls)
    image : np.ndarray
        the input image to run the tracker on
    """
     # Check if there are any detections
    if detections.size > 0:
        # Update tracker
        strongsort_tracker.update(detections, image, embeddings)
    else:
        detections = np.empty((0, 6))  # empty N X (x, y, x, y, conf, cls)
        strongsort_tracker.update(detections, image, embeddings)  # M X (x, y, x, y, id, conf, cls, ind)

    # Get tracking results and plot them on the image
    tracked_image = strongsort_tracker.plot_results(image, show_trajectories=True)
    
    return tracked_image


def main():
    # Create output directories if they don't exist
    YoloPaths.yolo_output_dir.mkdir(parents=True, exist_ok=True)
    # Step 1: Extract detections
    #logging.info("Starting detection process...")
    # for video_path in DetectionPaths.videos_input_dir.iterdir():  
    #     extract_detections(video_path, YoloPaths.yolo_output_dir)

    logging.info("Starting tracking process...")
    for video_path in DetectionPaths.videos_input_dir.iterdir():  
        # Step 2: Run tracking on extracted detections
        detections_path = YoloPaths.yolo_output_dir / f"{video_path.stem}_detections.npy"
        run_tracking_on_detections(detections_path, StrongSortPaths.deep_sort_output_dir, tracker='deep_oc_sort')
    logging.info("Process completed.")
    
    
if __name__ == "__main__":
    main()

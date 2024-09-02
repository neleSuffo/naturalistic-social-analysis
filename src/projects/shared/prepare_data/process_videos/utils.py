import logging
import cv2
import re
from pathlib import Path
from src.projects.social_interactions.common.constants import VideoParameters


def extract_frames_from_single_video(
    video_file: Path, 
    output_dir: Path, 
    fps: int
) -> bool:
    """
    This function extracts frames from a single video file and saves them as images in the output directory.

    Parameters
    ----------
    video_file : Path
        The path to the video file.
    output_dir : Path
        The directory to save the extracted frames.
    fps : int
        The frames per second to extract.
    
    Returns
    -------
    bool
        True if the frame extraction for all frames was successful, False otherwise.
    """
    logging.info(f"Starting frame extraction from video: {video_file} at {fps} FPS.")
    all_frames_success = True  # Flag to track success

    # Open the video file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        logging.error(f"Failed to open video file: {video_file}")
        return False
    
    # Get the frame rate and calculate the frame interval
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if frame_rate == 0:
        logging.error(f"Failed to get frame rate for video file: {video_file}")
        return False
        
    frame_interval = int(round(frame_rate / fps))
    nr_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_id in range(0, nr_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, image = cap.read()
        if success:
            try:
                # Save the frame as an image
                video_file_name = video_file.stem
                output_path = output_dir / f"{video_file_name}_{frame_id:06d}.jpg"
                cv2.imwrite(str(output_path), image)
            except Exception as e:
                logging.warning(f"Failed to save frame {frame_id} from {video_file}: {e}")
                all_frames_success = False
        else:
            logging.warning(f"Failed to read frame {frame_id} from {video_file}")
            all_frames_success = False
            
    cap.release()
    
    if all_frames_success:
        logging.info(f"Completed frame extraction for video: {video_file}")
        # Log success
        with open(VideoParameters.success_log_path, "a") as file:
            match = re.search(r'(id\d+.*)', str(video_file))
            if match:
                file.write(f"{match.group()}\n")
            else:
                file.write("Pattern not found\n")
    else:
        logging.warning(f"Frame extraction incomplete for video: {video_file}")

    return all_frames_success

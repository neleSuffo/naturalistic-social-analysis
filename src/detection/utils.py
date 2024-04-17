import numpy as np
import cv2
from typing import Tuple

def get_video_properties(video_path: str) -> Tuple[cv2.VideoCapture, int, int, int, int]:
    """
    This function loads a video from a given path and returns its properties.

    Parameters
    ----------
    video_path : str
        the path to the video file

    Returns
    -------
    Tuple[cv2.VideoCapture, int, int, int, int]
        the video capture object, the width of the frame, the height of the frame, the number of frames in the video, the frames per second
    """
    # Load the video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties of the video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_second = cap.get(cv2.CAP_PROP_FPS)

    return cap, frame_width, frame_height, frame_count, frames_per_second


def create_video_writer(output_path: str, 
                        frames_per_second: int,
                        frame_width: int,
                        frame_height: int,
                        nr_of_bars: int,
                        bar_height: int=20) -> cv2.VideoWriter:
    """
    This function creates a VideoWriter object to write the output video.

    Parameters
    ----------
    output_path : str
        the path to the output video file
    frames_per_second : int
        the frames per second of the video
    frame_width : int
        the width of the frame
    frame_height : int
        the height of the frame
    nr_of_bars : int
        the number of bars to concatenate
    bar_height : int, optional
        the height of each bar, by default 20

    Returns
    -------
    cv2.VideoWriter
        the video writer object
    """
    
    # Create a VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, frames_per_second, (frame_width, frame_height+(nr_of_bars*bar_height)))

    return out


def concat_video_with_bars(video_path: str, 
                           video_output_path: str,
                           persons_detection_bar: np.ndarray,
                           language_detection_bar: np.ndarray) -> None:
    """
    This function loads a video from a given path and concatenates the video with the given detection bars.

    Parameters
    ----------
    video_path : str
        the path to the video file
    video_output_path : str
        the path to the output video file
    persons_detection_bar : np.ndarray  
        the grey bar with green markers if a person is detected
    language_detection_bar : np.ndarray
        the grey bar with green markers if speech is audible
    """
    # Load video file and extract properties
    cap, frame_width, frame_height, frame_count, frames_per_second = get_video_properties(video_path)
    # Create a VideoWriter object to write the output video
    out = create_video_writer(video_output_path, frames_per_second, frame_width, frame_height, 2)  

    # Read the video frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Concatenate the frame and the grey bars
        frame_with_bars = cv2.vconcat([frame, persons_detection_bar, language_detection_bar])

        out.write(frame_with_bars)

    cap.release()
    out.release()
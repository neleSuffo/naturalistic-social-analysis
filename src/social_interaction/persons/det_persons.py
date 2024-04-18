from typing import Tuple
import cv2
import numpy as np
from PIL import Image
import torch
import sys
import os

# Get the grandparent directory of the current file
grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# Add the grandparent directory to the system path
sys.path.append(grandparent_dir)
# Import the utils module
from social_interaction import utils

def person_detection(video_input_path: str, 
                     video_output_path: str,
                     bar_height: int=20,
                     class_name: str='person') -> Tuple[np.ndarray, list]:
    """
    This function loads a video from a given path and creates a VideoWriter object to write the output video.
    It performs frame-wise person detection and adds a detection bar to the bottom of the frame. 
    If a person is detected in the frame, a green marker is added to the detection bar.

    Parameters
    ----------
    video_input_path : str
        the path to the video file
    video_output_path : str
        the path to the output video file
    bar_height : int, optional
        the height of the bar, by default 20
    class_name : str, optional
        the class name to detect, by default 'person'
    
    Returns 
    -------
    np.ndarray
        the detection bar with green markers if a person is detected
    list
        the results for each frame (1 if a person is detected, 0 otherwise)
    """
    # Load video file and extract properties
    cap, frame_width, frame_height, frame_count, frames_per_second = utils.get_video_properties(video_input_path)
    # Create a VideoWriter object to write the output video
    out = utils.create_video_writer(video_output_path, frames_per_second, frame_width, frame_height, 0) 

    detection_bar, detection_list = frame_wise_person_detection_with_bar(cap, 
                                                                         frame_width,
                                                                         frame_count, 
                                                                         out,
                                                                         bar_height,
                                                                         class_name=class_name)
    return detection_bar, detection_list

def frame_wise_person_detection_with_bar(cap: cv2.VideoCapture, 
                                         frame_width: int, 
                                         frame_count: int, 
                                         out: cv2.VideoWriter, 
                                         bar_height: int, 
                                         class_name: str='person') -> Tuple[np.ndarray, list]:
    """
    This function performs frame-wise person detection and adds a detection bar to the bottom of the frame.
    If a person is detected in the frame, a green marker is added to the detection bar.

    Parameters
    ----------
    cap : cv2.VideoCapture
        the video capture object
    frame_width : int
        the width of the frame
    frame_count : int
        the number of frames in the video
    out : cv2.VideoWriter
        the video writer object
    bar_height : int
        the height of the bar
    class_name : str, optional
        the class name to detect, by default 'person'

    Returns
    -------
    np.ndarray
        the detection bar with green markers if a person is detected
    list
        the results for each frame (1 if a person is detected, 0 otherwise)
    """
    # Create a detection bar equivalent to the length of the video
    detection_bar = np.full((bar_height, frame_width, 3), 128, dtype=np.uint8)

    # Initialize detection list to store detection results (1 if a person is detected, 0 otherwise)   
    detection_list = []

    # Load the YOLOv5 small model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    
    # Get the class labels
    class_list = model.names
    
    # Define the class name of interest and index
    class_index_det = [key for key, value in class_list.items() if value == class_name][0]
    
    # Loop through frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        # Convert frame to PIL Image
        img = Image.fromarray(frame[..., ::-1])
    
        # Apply object detection
        results = model(img)

        # Get the class index for every detected object per frame
        if len(results.pred[0]) > 0:
            detection_results_per_frame = [result[0][-1].item() for result in results.pred]
            # Check if the class index of interest is in the detection results
            detection_list.append(1 if class_index_det in detection_results_per_frame else 0)
        else:   
            detection_list.append(0)
    
        # Draw bounding boxes
        for result in results.pred:
            for det in result:
                x1, y1, x2, y2, conf, cls = det
                # Check if the detected class is equal to the class index of interest
                if int(cls) == class_index_det:
                    # Draw bounding box and label 
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (146,123,45), 2)
                    cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (146,123,45), 2)
                    # Draw a green marker on the detection bar if a person is detected
                    marker_position = int((cap.get(cv2.CAP_PROP_POS_FRAMES) / frame_count) * frame_width)
                    cv2.line(detection_bar, (marker_position, 0), (marker_position, bar_height), (146,123,45), thickness=2)
                
        # Write frame to output video
        out.write(frame)

        # Display modified frame
        # Only for testing purposes
        # cv2.imshow('Object Detection', frame_with_bar)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release video capture and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return detection_bar, detection_list
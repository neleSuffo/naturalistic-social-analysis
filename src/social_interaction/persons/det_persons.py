import cv2
from PIL import Image
import torch
import sys
import os
# Get the directory of my_utils.py
my_utils_dir = os.path.dirname(os.path.realpath('path/to/my_utils.py'))
# Add the directory to the Python path
sys.path.append(my_utils_dir)
# Now you can import my_utils
import my_utils

def person_detection(video_input_path: str, 
                     video_output_path: str,
                     class_name: str='person') -> list:
    """
    This function loads a video from a given path and creates a VideoWriter object to write the output video.
    It performs frame-wise person detection and returns the detection list (1 if a person is detected, 0 otherwise).

    Parameters
    ----------
    video_input_path : str
        the path to the video file
    video_output_path : str
        the path to the output video file
    class_name : str, optional
        the class name to detect, by default 'person'
    
    Returns 
    -------
    list
        the results for each frame (1 if a person is detected, 0 otherwise)
    """
    # Load video file and extract properties
    cap, frame_width, frame_height, frames_per_second = my_utils.get_video_properties(video_input_path)
    # Create a VideoWriter object to write the output video
    out = my_utils.create_video_writer(video_output_path, frames_per_second, frame_width, frame_height) 

    # Perform frame-wise detection
    detection_list = frame_wise_detection(cap, 
                                          out,
                                          class_name=class_name)
    return detection_list

def frame_wise_detection(cap: cv2.VideoCapture,
                         out: cv2.VideoWriter,
                         class_name: str) -> list:
    """
    This function performs frame-wise person detection on a video.
    It creates a detection list to store the detection results (1 if a person is detected, 0 otherwise).

    Parameters
    ----------
    cap : cv2.VideoCapture
        the video capture object
    out : cv2.VideoWriter
        the video writer object
    class_name : str, optional
        the class name to detect

    Returns
    -------
    list
        the results for each frame (1 if a person is detected, 0 otherwise)
    """

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
    return detection_list
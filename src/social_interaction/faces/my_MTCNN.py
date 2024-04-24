import os
import sys

import cv2
from facenet_pytorch import MTCNN

# Get the directory of my_utils.py
my_utils_dir = os.path.dirname(
    os.path.realpath(
        "/Users/nelesuffo/projects/leuphana-IPE/src/social_interaction/my_utils.py"  # noqa: E501
    )
)  # noqa: E501
# Add the directory to the Python path
sys.path.append(my_utils_dir)
# Now you can import my_utils
import my_utils  # noqa: E402


def face_detection(video_input_path: str, video_output_path: str) -> list:
    """
    This function loads a video from a given path and creates
    a VideoWriter object to write the output video.
    It performs frame-wise face detection and
    returns the detection list (1 if a face is detected, 0 otherwise).

    Parameters
    ----------
    video_input_path : str
        the path to the video file
    video_output_path : str
        the path to the output video file

    Returns
    -------
    list
        the results for each frame (1 if a face is detected, 0 otherwise)
    """
    # Load video file and extract properties
    (
        cap,
        frame_width,
        frame_height,
        frames_per_second,
    ) = my_utils.get_video_properties(  # noqa: E501
        video_input_path
    )  # noqa: E501
    # Create a VideoWriter object to write the output video
    out = my_utils.create_video_writer(
        video_output_path, frames_per_second, frame_width, frame_height
    )  # noqa: E501

    # Perform frame-wise detection
    detection_list = frame_wise_face_detection(cap, out)
    return detection_list


def frame_wise_face_detection(
    cap: cv2.VideoCapture, out: cv2.VideoWriter
) -> list:  # noqa: E501
    """
    This function performs frame-wise face detection on a video.
    It creates a detection list to store the detection results
    (1 if a face is detected, 0 otherwise).

    Parameters
    ----------
    cap : cv2.VideoCapture
        the video capture object
    out : cv2.VideoWriter
        the video writer object

    Returns
    -------
    list
        the results for each frame (1 if a face is detected, 0 otherwise)
    """
    # Initialize detection list to store detection results
    # (1 if a person is detected, 0 otherwise)
    detection_list = []

    # Initialize MTCNN face detector
    mtcnn = MTCNN()

    # Iterate over frames in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB (MTCNN expects RGB format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        boxes, _ = mtcnn.detect(rgb_frame)

        # Draw bounding boxes around detected faces
        if boxes is not None:
            # Append 1 to the detection list if a face is detected
            detection_list.append(1)
            for box in boxes:
                x, y, w, h = box.astype(int)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            # Append 0 to the detection list if no face is detected
            detection_list.append(0)

        # Write frame to output video
        out.write(frame)

        # Display the frame with detected faces
        # Only for visualization purposes
        # cv2.imshow('Video', frame)
        # Press 'q' to exit
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #   break

    # Release video capture and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return detection_list

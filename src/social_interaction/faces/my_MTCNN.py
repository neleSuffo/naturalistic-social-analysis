import cv2
from facenet_pytorch import MTCNN
from faces import my_utils


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
    # Initialize MTCNN face detector
    mtcnn = MTCNN()

    # Load video and get video properties
    cap = cv2.VideoCapture(video_input_path)
    try:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = int(cap.get(cv2.CAP_PROP_FPS))

        # Create a VideoWriter object to write the output video
        out = my_utils.create_video_writer(
            video_output_path, frames_per_second, frame_width, frame_height
        )
        try:
            # Perform frame-wise detection
            detection_list = frame_wise_face_detection(mtcnn, cap, out)
            return detection_list
        finally:
            out.release()
    finally:
        cap.release()

    return detection_list


def frame_wise_face_detection(
    mtcnn: MTCNN, cap: cv2.VideoCapture, out: cv2.VideoWriter
) -> list:
    """
    This function performs frame-wise face detection on a video.
    It creates a detection list to store the detection results
    (1 if a face is detected, 0 otherwise).

    Parameters
    ----------
    mtcnn : MTCNN
        the MTCNN face detector
    cap : cv2.VideoCapture
        the video capture object
    out : cv2.VideoWriter
        the video writer object

    Returns
    -------
    list
        the results for each frame
        (1 if a face is detected, 0 otherwise)
    """
    # Initialize detection list to store detection results
    # (1 if a person is detected, 0 otherwise)
    detection_list = []

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

    return detection_list

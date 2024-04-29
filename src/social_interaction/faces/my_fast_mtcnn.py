from facenet_pytorch import MTCNN
import cv2

filenames = "/Users/nelesuffo/projects/leuphana-IPE/data/sample_2.MP4"


def run_face_detection(video_input_path: str, model: MTCNN) -> list:
    """
    This function loads a video from a given path and creates
    a VideoWriter object to write the output video.
    It performs face detection in batch mode and returns the results
    (1 if a face is detected, 0 otherwise)

    Parameters
    ----------
    video_input_path : str
        the path to the video file
    model : MTCNN
        the MTCNN face detector

    Returns
    -------
    list
        the results for each frame (1 if a face is detected, 0 otherwise)
    """
    # Load video and get video properties
    cap = cv2.VideoCapture(video_input_path)
    try:
        # Perform batch frame face detection
        detection_list = batch_frame_face_detection(model, cap)
        return detection_list
    finally:
        cap.release()


def batch_frame_face_detection(model: MTCNN, cap: cv2.VideoCapture) -> list:
    """
    This function performs batch frame face detection on a video.

    Parameters
    ----------
    model : MTCNN
        the MTCNN face detector
    cap : cv2.VideoCapture
        the video capture object

    Returns
    -------
    list
        the results for the batch of frames
        (1 if a face is detected, 0 otherwise)
    """
    # Initialize detection list to store detection results
    # (1 if a person is detected, 0 otherwise)
    detection_list = []
    frames = []
    batch_size = 4

    # Get the number of frames in the video
    nr_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_num in range(nr_frames):
        # Load frame
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Append frame to batch
        frames.append(frame)

        # Perform face detection for the batch of frames
        if len(frames) >= batch_size or frame_num == nr_frames - 1:
            # Perform face detection on the batch of frames (returns batch/stride number of boxes and probabilities)
            batch_detection_list = model(frames)
            # Append detection batch results to detection list
            detection_list.extend(batch_detection_list)
            frames = []
    return detection_list

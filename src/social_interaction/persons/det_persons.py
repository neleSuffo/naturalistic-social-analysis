from moviepy.editor import VideoFileClip
from src.social_interaction.constants import DetectionOutputPaths, DetectionParameters
from config import generate_detection_output_video
from persons import my_utils
import cv2
import os
import torch


def run_person_detection(
    video_input_path: str,
    video_file_name: str,
    model: torch.nn.Module,
    detection_output: dict,
) -> dict:
    """
    This function loads a video from a given path and
    creates a VideoWriter object to write the output video.
    It performs frame-wise person detection and returns
    the detection list (1 if a person is detected, 0 otherwise).

    Parameters
    ----------
    video_input_path : str
        the path to the video file
    video_file_name : str
        the name of the video file
    model : torch.nn.Module
        the YOLOv5 model
    detection_output : dict
        the detection output dictionary in COCO format

    Returns
    -------
    dict
        the detection results in COCO format
    """
    # Get the class labels and class index
    class_list = model.names
    class_index_det = [
        key
        for key, value in class_list.items()
        if value == DetectionParameters.yolo_detection_class
    ][0]

    # Load video file
    cap = cv2.VideoCapture(video_input_path)

    # If a detection output video should be generated
    if generate_detection_output_video:
        # Load video file and extract properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = int(cap.get(cv2.CAP_PROP_FPS))

        # Create video output directory if it does not exist
        video_output_path = os.path.join(DetectionOutputPaths.person, video_file_name)

        # Create a VideoWriter object to write the output video
        out = my_utils.create_video_writer(
            video_output_path, frames_per_second, frame_width, frame_height
        )
        # Perform detection and write output video
        person_detection_video_output(
            cap,
            out,
            video_input_path,
            video_output_path,
            model,
            class_index_det,
        )
    else:
        # Perform detection and return results in
        detection_output_filled = person_detection_coco_output(
            cap,
            model,
            class_index_det,
            detection_output,
        )

        return detection_output_filled


def person_detection_video_output(
    cap: cv2.VideoCapture,
    out: cv2.VideoWriter,
    video_input_path: str,
    video_output_path: str,
    model: torch.nn.Module,
    class_index_det: int,
) -> None:
    """
    This function performs frame-wise person detection on a video file (every frame_step-th frame)
    It draws bounding boxes around detected persons and writes the output video.

    Parameters
    ----------
    cap : cv2.VideoCapture
        the video capture object
    out : cv2.VideoWriter
        the video writer object
    video_input_path: str
        the path to the video file
    video_output_path : str
        the path to the output video file
    model : torch.nn.Module
        the YOLOv5 model
    class_index_det : int
        the class index of the class to detect
    """
    # Initialize frame count
    frame_count = 0

    # Initialize detection results
    detections = []

    # Loop through frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Increment frame count
        frame_count += 1

        # Perform detection every frame_step-th frame
        if frame_count % DetectionParameters.frame_step == 0:
            # Apply object detection
            results = model(frame)

            # Clear previous detections
            detections.clear()

            # Store new detections
            for result in results.pred:
                for det in result:
                    x1, y1, x2, y2, conf, cls = det
                    # Check if the detected class is equal
                    # to the class index of interest
                    if int(cls) == class_index_det:
                        detections.append((x1, y1, x2, y2, conf, cls))

        # Draw bounding boxes from the last detection
        for x1, y1, x2, y2, conf, cls in detections:
            # Draw bounding box and label
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (146, 123, 45),
                2,
            )
            cv2.putText(
                frame,
                f"{model.names[int(cls)]} {conf:.2f}",  # noqa: E501, E231
                (int(x1), int(y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (146, 123, 45),
                2,
            )

        # Write frame to output video
        out.write(frame)

    # Release video capture and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Add audio to the output video
    clip = VideoFileClip(video_output_path)

    # Get the filename and extension
    filename, file_extension = os.path.splitext(video_output_path)
    processed_filename = f"{filename}_processed{file_extension}"

    # Write the video file with audio
    clip.write_videofile(processed_filename, codec="libx264", audio=video_input_path)

    # Delete the video file without audio
    os.remove(video_output_path)


def person_detection_coco_output(
    cap: cv2.VideoCapture,
    model: torch.nn.Module,
    class_index_det: int,
    detection_output: dict,
) -> dict:
    """
    This function performs person detection on a video file (every frame_step-th frame)
    It returns the detection results in COCO format.

    Parameters
    ----------
    cap : cv2.VideoCapture
        the video capture object
    model : torch.nn.Module
        the YOLOv5 model
    class_index_det : int
        the class index of the class to detect
    detection_output : dict
        the detection output dictionary in COCO format
    Returns
    -------
    dict
        the detection results in COCO format
    """
    # Initialize frame count
    frame_count = 0

    # Initialize annotation id
    annotation_id = 0

    # Loop through frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Increment frame count
        frame_count += 1

        # Perform detection every frame_step-th frame
        if frame_count % DetectionParameters.frame_step == 0:
            # Apply object detection
            results = model(frame)

            # Add image information to COCO output
            detection_output["images"].append(
                {"id": frame_count, "width": frame.shape[1], "height": frame.shape[0]}
            )

            # Add detection results to COCO output
            for result in results.pred:
                for det in result:
                    x1, y1, x2, y2, conf, cls = det
                    # Check if the detected class is equal
                    # to the class index of interest
                    if int(cls) == class_index_det:
                        # Add annotation information to COCO output
                        detection_output["annotations"].append(
                            {
                                "id": annotation_id,
                                "image_id": frame_count,
                                "category_id": class_index_det,
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "score": conf,
                            }
                        )
                        annotation_id += 1

    return detection_output

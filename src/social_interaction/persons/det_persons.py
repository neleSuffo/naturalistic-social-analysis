from typing import Generator
import cv2
import torch
from persons import my_utils


def run_person_detection(
    video_input_path: str,
    video_output_path: str,
    model: torch.nn.Module,
    class_name: str = "person",
) -> list:
    """
    This function loads a video from a given path and
    creates a VideoWriter object to write the output video.
    It performs frame-wise person detection and returns
    the detection list (1 if a person is detected, 0 otherwise).

    Parameters
    ----------
    video_input_path : str
        the path to the video file
    video_output_path : str
        the path to the output video file
    model : torch.nn.Module
        the YOLOv5 model
    class_name : str, optional
        the class name to detect, by default 'person'

    Returns
    -------
    list
        the results for each frame (1 if a person is detected, 0 otherwise)
    """
    # Load video file and extract properties
    cap = cv2.VideoCapture(video_input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = int(cap.get(cv2.CAP_PROP_FPS))

    # Create a VideoWriter object to write the output video
    out = my_utils.create_video_writer(
        video_output_path, frames_per_second, frame_width, frame_height
    )

    # Get the class labels and class index
    class_list = model.names
    class_index_det = [key for key, value in class_list.items() if value == class_name][
        0
    ]

    # Perform frame-wise detection
    detection_list = list(frame_wise_person_detection(cap, out, model, class_index_det))
    return detection_list


def frame_wise_person_detection(
    cap: cv2.VideoCapture,
    out: cv2.VideoWriter,
    model: torch.nn.Module,
    class_index_det: int,
) -> Generator[int, None, None]:
    """
    This function performs frame-wise person detection on a video.
    It creates a detection list to store the detection results
    (1 if a person is detected, 0 otherwise).

    Parameters
    ----------
    cap : cv2.VideoCapture
        the video capture object
    out : cv2.VideoWriter
        the video writer object
    model : torch.nn.Module
        the YOLOv5 model
    class_index_det : int
        the class index of the class to detect

    Returns
    -------
    Generator[int, None, None]
        the detection list (1 if a person is detected, 0 otherwise)

    """
    # Loop through frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply object detection
        results = model(frame)

        # Get the class index for every detected object per frame
        if len(results.pred[0]) > 0:
            detection_results_per_frame = [
                result[0][-1].item() for result in results.pred
            ]
            # Check if the class index of interest is in the detection results
            yield 1 if class_index_det in detection_results_per_frame else 0
        else:
            yield 0

        # Draw bounding boxes
        for result in results.pred:
            for det in result:
                x1, y1, x2, y2, conf, cls = det
                # Check if the detected class is equal
                # to the class index of interest
                if int(cls) == class_index_det:
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

        # Display modified frame
        # Only for testing purposes
        # cv2.imshow('Object Detection', frame_with_bar)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release video capture and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

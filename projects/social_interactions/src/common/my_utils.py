from projects.social_interactions.src.common.constants import DetectionParameters, LabelToCategoryMapping
from facenet_pytorch import MTCNN
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import os
import json
import logging
import cv2


# Function to extract frames from a video
def extract_frames(
    video_path: str, 
    output_dir: str
) -> list:
    """
    This function extracts frames from a video file and saves them to a directory.
    It then returns the list of extracted frames.

    Parameters
    ----------
    video_path : str
        the path to the video file
    output_dir : str
        the directory to save the extracted frames

    Returns
    -------
    list
        the list of extracted frame paths
    """
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    # Loop through frames
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frame_name = f'{os.path.basename(video_path).split(".")[0]}_frame_{i:06d}.jpg'
        frame_path = os.path.join(output_dir, frame_name)
        # Save frame
        cv2.imwrite(frame_path, frame)
        # Append frame path to list
        frames.append(frame_path)

    cap.release()
    return frames


# TODO: Add proximity detection
def run_proximity_detection():
    pass  # Implement proximity detection here


def create_video_writer(
    output_path: str,
    frames_per_second: int,
    frame_width: int,
    frame_height: int,
) -> cv2.VideoWriter:
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

    Returns
    -------
    cv2.VideoWriter
        the video writer object
    """

    # Create a VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path, fourcc, frames_per_second, (frame_width, frame_height)
    )

    return out


def detection_video_output(
    cap: cv2.VideoCapture,
    out: cv2.VideoWriter,
    video_input_path: str,
    video_output_path: str,
    model,
    detection_fn,
    draw_fn,
    class_index_det: int = None,
) -> None:
    """
    This function performs frame-wise detection on a video file (every frame_step-th frame)
    It draws bounding boxes around detected objects and writes the output video.

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
    model
        the detection model
    detection_fn : function
        the function to perform detection
    draw_fn : function
        the function to draw bounding boxes
    """
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a progress bar
    with tqdm(total=total_frames, desc="Processing frames", ncols=70) as pbar:
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

            # Update progress bar
            pbar.update()

            # Perform detection every frame_step-th frame
            if frame_count % DetectionParameters.frame_step == 0:
                # Apply object detection
                detections = detection_fn(model, frame, class_index_det)

            # Draw bounding boxes from the last detection
            for detection in detections:
                draw_fn(frame, detection)

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
        clip.write_videofile(
            processed_filename, codec="libx264", audio=video_input_path
        )

        # Delete the video file without audio
        os.remove(video_output_path)


def detection_coco_output(
    cap: cv2.VideoCapture,
    model,
    detection_fn,
    process_results_fn,
    video_file_name: str,
    file_id: str,
    annotation_id: int,
    class_index_det: int = None,
) -> dict:
    """
    This function performs detection on a video file (every frame_step-th frame)
    It returns the detection results in COCO format.

    Parameters
    ----------
    cap : cv2.VideoCapture
        the video capture object
    model
        the model used for detection
    detection_fn : callable
        the function used to perform detection
    process_results_fn : callable
        the function used to process the detection results
    video_file_name: str
        the name of the video file
    file_id: str
        the video file id
    annotation_id: int
        the annotation id
    class_index_det : int
        the class index of the class to detect (only for yolo detection), defaults to None

    Returns
    -------
    dict
        the detection results in COCO format
    """
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a progress bar
    with tqdm(total=total_frames, desc="Processing frames", ncols=70) as pbar:
        # Initialize detection output
        detection_output = {
            "images": [],
            "annotations": [],
        }

        if class_index_det is None:
            # Get the category id for face detection
            category_id = LabelToCategoryMapping.label_dict[
                DetectionParameters.mtcnn_detection_class
            ]
        else:
            # Get the category ID for person detection
            category_id = LabelToCategoryMapping.label_dict[
                DetectionParameters.yolo_detection_class
            ]

        # Initialize frame count, annotation id and image id
        frame_count = 0
        image_id = 0

        # Loop through frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Increment frame count
            frame_count += 1

            # Update progress bar
            pbar.update()

            # Perform detection every frame_step-th frame
            if frame_count % DetectionParameters.frame_step == 0:
                # Apply object detection
                results = detection_fn(model, frame)

                # Add image information to COCO output
                detection_output["images"].append(
                    {
                        "id": image_id,
                        "video_id": file_id,
                        "frame_id": frame_count,
                        "file_name": f"{video_file_name}_{frame_count}.jpg",
                    }
                )
                image_id += 1
                # Process detection results and add to COCO output
                process_results_fn(
                    results,
                    detection_output,
                    frame_count,
                    category_id,
                    annotation_id,
                    class_index_det,
                )

        return detection_output


def create_video_to_id_mapping(video_names: list) -> dict:
    """
    This function creates a dictionary with mappings from video names to ids.

    Parameters
    ----------
    video_names : list
        a list of video names

    Returns
    -------
    dict
        a dictionary with mappings from video names to ids
    """
    video_id_dict = {video_name: i for i, video_name in enumerate(video_names)}
    return video_id_dict


def save_results_to_json(results: dict, output_path: str) -> None:
    """
    This function saves the results to a JSON file.

    Parameters
    ----------
    results : dict
        the results for each detection type
    output_path : str
        the path to the output file
    """
    directory = os.path.dirname(output_path)
    # Check if the output directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Save the results to a JSON file
    with open(output_path, "w") as f:
        json.dump(results, f)


def display_results(results: dict) -> None:
    """
    This function calculates the percentage of frames where the object is detected
    and prints the results.

    Parameters
    ----------
    results : dict
        the results for each detection type
    """
    for detection_type, detection_list in results.items():
        percentage = sum(detection_list) / len(detection_list) * 100
        logging.info(
            f"Percentages of at least one {detection_type} detected relative to the total number of frames: {percentage:.2f}"
        )
        logging.info(
            f"Total number of steps (every {DetectionParameters.frame_step}-th frame) ({detection_type}): {len(detection_list)}"
        )

from src.projects.social_interactions.common.constants import DetectionParameters, LabelToCategoryMapping, VTCParameters
from moviepy.editor import VideoFileClip
from pathlib import Path
import tempfile
from tqdm import tqdm
import json
import logging
import cv2
import subprocess
import multiprocessing

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
        frame_name = f'{video_path.stem}_frame_{i:06d}.jpg'
        frame_path = Path(output_dir) / frame_name        # Save frame
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
        processed_filename = f"{video_output_path.stem}_processed{video_output_path.suffix}"
        # Write the video file with audio
        clip.write_videofile(
            processed_filename, codec="libx264", audio=video_input_path
        )

        # Delete the video file without audio
        video_output_path.unlink()


def detection_coco_output(
    cap: cv2.VideoCapture,
    model,
    detection_fn,
    process_results_fn,
    video_file_name: str,
    file_id: str,
    annotation_id: multiprocessing.Value,
    image_id: multiprocessing.Value,
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
    annotation_id: multiprocessing.Value
        the unique annotation id
    image_id: multiprocessing.Value
        the unique image id
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

        # Initialize frame count and image id
        frame_count = 0

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
                        "id": image_id.value,
                        "video_id": file_id,
                        "frame_id": frame_count,
                        "file_name": f"{video_file_name}_{frame_count}.jpg",
                    }
                )
                # Process detection results and add to COCO output
                process_results_fn(
                    results,
                    detection_output,
                    category_id,
                    annotation_id,
                    image_id,
                    class_index_det,
                )
                # Increment image id
                with image_id.get_lock():
                    image_id.value += 1

        return detection_output


def create_video_to_id_mapping(video_names: list) -> dict:
    """
    This function creates a dictionary with mappings from video names to ids.

    Parameters
    ----------
    video_names : list
        a list of video names without the file extension

    Returns
    -------
    dict
        a dictionary with mappings from video names to ids
    """
    # Create a dictionary with mappings from video names to ids
    # the first video has id 0, the second video has id 1, and so on
    video_id_dict = {video_name: i for i, video_name in enumerate(video_names)}
    return video_id_dict


def update_or_create_json_file(
    path: Path,
    new_data: dict,
    existing_data: dict = None,
) -> None:
    """
    Updates an existing JSON file with new data or creates a new file if it doesn't exist.

    Parameters
    ----------
    path : Path
        the path to the JSON file
    new_data : dict
        the new data to add or create the file with
    existing_data : dict
        the existing data to update, if any. If None, the function will load existing data from the file if it exists
    """
    # Check if the file exists
    if path.exists() and existing_data is None:
        # File exists and no existing data provided, load the existing data
        with path.open('r') as file:
            existing_data = json.load(file)

    # If existing data was provided or loaded, update it with the new data
    if existing_data is not None:
        existing_data.update(new_data)
        data_to_write = existing_data
    else:
        # No existing data, use new data as is
        data_to_write = new_data

    # Write the data back to the file
    with path.open('w') as file:
        json.dump(data_to_write, file, indent=4)


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


def extract_audio_from_video(video: VideoFileClip, filename: str) -> None:
    """
    This function extracts the audio from a video file
    and saves it as a 16kHz WAV file.

    Parameters
    ----------
    video : VideoFileClip
        the video file
    filename : str
        the filename of the video
    """
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=True)

    # Extract the audio and save it to the temporary file
    video.audio.write_audiofile(temp_file.name + ".wav", codec="pcm_s16le")

    # Create the output directory if it doesn't exist
    VTCParameters.audio_path.mkdir(parents=True, exist_ok=True)

    # Convert the audio to 16kHz with sox and
    # save it to the output file
    output_file = VTCParameters.audio_path / f"{filename}{VTCParameters.audio_name_ending}"

    subprocess.run(
        ["sox", temp_file.name + ".wav", "-r", "16000", output_file],
        check=True,
    )
    
    # Delete the temporary file
    temp_file.close()
    
    logging.info(f"Successfully stored the file at {output_file}")


def extract_audio_from_videos_in_folder(folder_path: Path) -> None:
    """
    Extracts audio from all video files in the specified folder, if not already done.
    """
    for video_file in folder_path.iterdir():
        if video_file.suffix.lower() not in ['.mp4']:
            continue  # Skip non-video files
        
        audio_path = VTCParameters.audio_path / f"{video_file.stem}{VTCParameters.audio_name_ending}"
        
        # Check if the audio file already exists
        if not audio_path.exists():
            # Create a VideoFileClip object
            video_clip = VideoFileClip(str(video_file))  
            # Extract audio from the video
            extract_audio_from_video(video_clip, video_file.stem)  
            print(f"Extracted audio from {video_file.name}")
        else:
            print(f"Audio already exists for {video_file.name}")

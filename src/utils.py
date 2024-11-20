import cv2
import logging
import subprocess
import sqlite3
import random
import shutil
import concurrent.futures
from pathlib import Path
from typing import List, Optional, Union, Tuple
import tempfile
from moviepy.editor import VideoFileClip
from constants import (
    DetectionPaths,
    ModelNames,
    VTCPaths
)
from config import (
    VideoConfig,
    StrongSortConfig,
    TrainingConfig,
    VTCConfig,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("extract_frames.log"),
        logging.StreamHandler()
    ]
)

def fetch_all_annotations(
    category_ids: Optional[List[int]] = None,
) -> List[tuple]:
    """
    This function fetches all annotations from the database (excluding the -1 category id)
    Categories with id -1 are labeled as "ignore" and are not included in the annotations.

    Parameters
    ----------
    category_ids : list, optional
        the list of category ids to filter the annotations,
        by default None


    Returns
    -------
    list of tuple
        the list of annotations
        (image_id, video_id, category_id, bbox, image_file_name, video_file_name)
    """
    conn = sqlite3.connect(DetectionPaths.annotations_db_path)
    cursor = conn.cursor()

    if category_ids:
        # Generate a string of ? placeholders that matches the length of category_ids
        placeholders = ", ".join("?" for _ in category_ids)

    if category_ids:
        # Generate a string of ? placeholders that matches the length of category_ids
        # Only fetch annotations that are not labeled noise (-1) and are not outside the frame (outside = 0)
        placeholders = ", ".join("?" for _ in category_ids)
        query = f"""
        SELECT DISTINCT 
            a.image_id, 
            a.video_id, 
            a.category_id, 
            a.bbox, 
            i.file_name, 
            v.file_name as video_file_name
        FROM annotations a
        JOIN images i ON a.image_id = i.frame_id AND a.video_id = i.video_id
        JOIN videos v ON a.video_id = v.id
        WHERE a.category_id IN ({placeholders}) 
            AND a.category_id != -1 
            AND a.outside = 0
            #TODO: comment out later
            AND a.video_id IN (129, 3, 18)
        ORDER BY a.video_id, a.image_id
        """
        cursor.execute(query, category_ids)
    else:
        query = """
        SELECT DISTINCT 
            a.image_id, 
            a.video_id, 
            a.category_id, 
            a.bbox, 
            i.file_name, 
            v.file_name as video_file_name
        FROM annotations a
        JOIN images i ON a.image_id = i.frame_id AND a.video_id = i.video_id
        JOIN videos v ON a.video_id = v.id
        WHERE a.category_id != -1 
            AND a.outside = 0
        ORDER BY a.video_id, a.image_id
        """
        cursor.execute(query)

    annotations = cursor.fetchall()
    conn.close()
    return annotations


def split_videos_into_train_val(
    input_folder: Path,
    ) -> Union[list, list]:
    """
    This function splits the videos in the input folder into train and validation sets
    It returns the list of video names in the train and validation sets.
    """
    # Get all video files in the input folder
    input_folder = DetectionPaths.videos_input_dir
    train_ratio = TrainingConfig.train_test_split_ratio
    video_files = [f for f in input_folder.iterdir() if f.is_file() and f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']]

    # Calculate total duration of all videos
    total_duration = sum(get_video_length(video) for video in video_files)
    train_duration_target = total_duration * train_ratio

    # Shuffle videos to randomize assignment
    random.shuffle(video_files)

    # Initialize durations
    current_train_duration = 0
    train_videos = []
    val_videos = []

    # Assign videos to train and val
    for video in video_files:
        video_duration = get_video_length(video)
        if current_train_duration + video_duration <= train_duration_target:
            train_videos.append(video.name)
            current_train_duration += video_duration
        else:
            val_videos.append(video.name)

    logging.info(f"Total duration: {total_duration:.2f} seconds")
    logging.info(f"Train set duration: {current_train_duration:.2f} seconds")
    logging.info(f"Validation set duration: {total_duration - current_train_duration:.2f} seconds")
    logging.info(f"Train videos: {len(train_videos)}")
    logging.info(f"Validation videos: {len(val_videos)}")

    return train_videos, val_videos


def get_frame_width_height(video_file_path: Path) -> tuple:
    """
    This function gets the frame width and height of a video file.

    Parameters
    ----------
    video_file_path : Path
        the path to the video file

    Returns
    -------
    tuple
        the frame width and height
    """

    # Open the video file
    cap = cv2.VideoCapture(video_file_path)

    # Get the frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Always release the VideoCapture object
    cap.release()
    return frame_width, frame_height


def get_video_length(
    file_path: Path
) -> float:
    """
    This function returns the length of a video file in seconds.
    
    Parameters
    ----------
    file_path : Path
        the path to the video file
    
    Returns
    -------
    float
        the length of the video in seconds
    """
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    return float(result.stdout)


def get_video_frame_count(
    video_path: Path
) -> int:
    """
    Get the total number of frames in a video.
    
    Parameters
    ----------
    video_path : str
        The path to the video file.
        
    Returns
    -------
    int
        The total number of frames in the video.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def split_videos(
    video_files: List[Path], 
    split_ratio: float
) -> Tuple[List[Path], List[Path]]:
    """
    Splits the list of videos into training and validation sets, balancing the split by video length.

    Parameters:
    ----------
    video_files: list
        The list of video files to split.
    split_ratio: float
        The ratio of training videos to validation videos.

    Returns:
    -------
    tuple of lists
        A tuple containing two lists: training videos and validation videos.
    """
    # Get the duration of each video
    video_durations = [(video_file, get_video_length(video_file)) for video_file in video_files]
    # Calculate the total duration of all videos
    total_duration = sum(duration for _, duration in video_durations)
    
    train_videos = []
    val_videos = []
    train_duration = 0
    val_duration = 0
    
    # Shuffle the videos to randomize the assignment
    random.shuffle(video_durations)

    # Assign videos to train and val sets
    for video_file, duration in video_durations:
        if train_duration / total_duration < split_ratio:
            train_videos.append(video_file)
            train_duration += duration
        else:
            val_videos.append(video_file)
            val_duration += duration

    logging.info(f"Total training duration: {train_duration} seconds")
    logging.info(f"Total validation duration: {val_duration} seconds")

    return train_videos, val_videos


def prepare_video_dataset(
    output_dir: Path,
    model: str,
    fps: int = None,
    batch_size: int = VideoConfig.video_batch_size,
) -> None:
    """
    Extracts frames from all videos in the given directory, splits them into training and validation sets,
    and saves the frames in corresponding folders.
    
    Parameters:
    ----------
    output_dir: Path
        The directory to save the extracted frames.
    split_ratio: float
        The ratio of training videos to validation videos.
    fps: int
        The frames per second to extract.
    model: str
        The model for which the frames are extracted.
    batch_size: int
        The number of videos to process concurrently.
    """
    logging.info(f"Starting frame extraction from videos in {video_dir} to {output_dir} at {fps} FPS with split ratio {split_ratio}.")

    video_dir = DetectionPaths.videos_input_dir
    split_ratio = TrainingConfig.train_test_split
    
    # Ensure output directories exist
    train_output_dir = output_dir / 'train'
    val_output_dir = output_dir / 'val'
    for dir_path in [train_output_dir, val_output_dir]:
        shutil.rmtree(dir_path, ignore_errors=True)
        dir_path.mkdir(parents=True, exist_ok=True)

    # Get all video files
    video_files = [f for f in video_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.mp4', '.MP4']]
    logging.info(f"Found {len(video_files)} video files to process.")

    # Split videos into train and val sets
    train_videos, val_videos = split_videos(video_files, split_ratio)

    # Process videos in batches
    for i in range(0, len(train_videos), batch_size):
        batch = train_videos[i:i + batch_size]
        logging.info(f"Processing training batch {i // batch_size + 1}")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            if model == ModelNames.yolo_model:
                executor.map(lambda video: process_video_yolo(video, train_output_dir), batch)
            if model == ModelNames.strong_sort_model:
                executor.map(lambda video: process_video_strong_sort(video, train_output_dir), batch)
                
    for i in range(0, len(val_videos), batch_size):
        batch = val_videos[i:i + batch_size]
        logging.info(f"Processing validation batch {i // batch_size + 1}")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            if model == ModelNames.yolo_model:
                executor.map(lambda video: process_video_yolo(video, train_output_dir), batch)
            if model == ModelNames.strong_sort_model:
                executor.map(lambda video: process_video_strong_sort(video, train_output_dir), batch)
    logging.info("Completed frame extraction for all videos.")


# Function to extract frames from a video
def extract_frames_from_video(
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


def process_video_strong_sort(video_file: Path, output_subdir: Path) -> None:
        """
        Extracts frames from a single video file and saves them to the output directory.

        Parameters:
        ----------
        video_file : Path
            The path to the video file.
        output_subdir : Path
            The subdirectory to save the extracted frames.
        """
        logging.info(f"Extracting frames from video: {video_file}")
        # Open the video file
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            logging.error(f"Failed to open video file: {video_file}")
            return

        # Create the output directory
        output_video_folder = output_subdir / video_file.stem / StrongSortConfig.image_subdir
        output_video_folder.mkdir(parents=True, exist_ok=True)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Increment frame count and format the file name
            frame_filename = f"{frame_count:06}.jpg"
            frame_filepath = output_video_folder / frame_filename
            cv2.imwrite(str(frame_filepath), frame)

            frame_count += 1

        cap.release()
        logging.info(f"Completed frame extraction for {video_file.name}.")


def process_video_yolo(
    video_file: Path, 
    output_dir: Path,
    fps: int
    ) -> None:
        """
        Extracts frames from a single video file and saves them to the output directory.

        Parameters:
        ----------
        video_file : Path
            The path to the video file.
        output_dir : Path
            The path to the output directory.
        fps : int
            The frames per second to extract.
        """
        logging.info(f"Extracting frames from video: {video_file}")
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            logging.error(f"Failed to open video file: {video_file}")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(round(frame_rate / fps))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_filename = f"{frame_count // frame_interval:06}.jpg"
                frame_filepath = output_dir / frame_filename
                cv2.imwrite(str(frame_filepath), frame)

            frame_count += 1

        cap.release()
        logging.info(f"Completed frame extraction for {video_file.name}.")
        
        
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
    VTCPaths.output_dir.mkdir(parents=True, exist_ok=True)

    # Convert the audio to 16kHz with sox and
    # save it to the output file
    output_file = VTCPaths.output_dir / f"{filename}{VTCConfig.audio_file_suffix}"

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
        if video_file.suffix.lower() not in ['.mp4', '.MP4']:
            continue  # Skip non-video files
        
        audio_path = VTCPaths.output_dir / f"{video_file.stem}{VTCConfig.audio_file_suffix}"
        
        # Check if the audio file already exists
        if not audio_path.exists():
            # Create a VideoFileClip object
            video_clip = VideoFileClip(str(video_file))  
            # Extract audio from the video
            extract_audio_from_video(video_clip, video_file.stem)  
            print(f"Extracted audio from {video_file.name}")
        else:
            print(f"Audio already exists for {video_file.name}")
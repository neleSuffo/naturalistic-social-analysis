import cv2
import os
import logging
import subprocess
import sqlite3
import random
import shutil
import gc  # Garbage collection
import concurrent.futures
from pathlib import Path
from typing import List, Optional, Union, Tuple
from tqdm import tqdm
import glob
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
    DetectionParameters,
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
    category_ids: List[int],
) -> List[tuple]:
    """
    This function fetches annotations from the database for specific category IDs.
    Supports fetching person annotations, object annotations, or both.

    Parameters
    ----------
    category_ids : List[int]
        The list of category IDs to filter the annotations
    
    Returns
    -------
    list of tuple
        The list of annotations.
    """
    logging.info(f"Fetching annotations for category IDs: {category_ids}")
    conn = sqlite3.connect(DetectionPaths.quantex_annotations_db_path)
    cursor = conn.cursor()
    
    placeholders = ", ".join("?" for _ in category_ids)
    object_target_class_ids = [3, 4, 5, 6, 7, 8, 12]

    # Construct conditional filter for object_interaction
    object_placeholders = ", ".join(str(x) for x in object_target_class_ids)
    
    query = f"""
    SELECT DISTINCT 
        a.category_id, 
        a.bbox, 
        a.object_interaction,
        i.file_name,
        a.gaze_directed_at_child,
        a.person_age
    FROM annotations a
    JOIN images i ON a.image_id = i.frame_id AND a.video_id = i.video_id
    JOIN videos v ON a.video_id = v.id
    WHERE a.category_id IN ({placeholders}) 
        AND a.outside = 0 
        AND v.file_name NOT LIKE '%id255237_2022_05_08_04%'
        AND (
            -- Only apply the interaction filter if category_id is an object category
            (a.category_id IN ({object_placeholders}) AND a.object_interaction = 'Yes') OR
            (a.category_id NOT IN ({object_placeholders}))
        )
    ORDER BY a.video_id, a.image_id
    """
    
    cursor.execute(query, category_ids)
    annotations = cursor.fetchall()
    conn.close()
    # log all unique category ids
    unique_category_ids = set(annotation[0] for annotation in annotations)
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
    logging.info(f"Starting frame extraction from videos in {video_dir} to {routput_dir} at {fps} FPS with split ratio {split_ratio}.")

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

def get_processed_videos(processed_videos_file: Path) -> set:
    """Read the list of already processed videos."""
    if not processed_videos_file.exists():
        return set()
    
    with processed_videos_file.open('r') as f:
        return set(line.strip() for line in f)

def mark_video_as_processed(video_path: Path, processed_videos_file: Path):
    """Mark a video as processed by adding it to the tracking file."""
    with processed_videos_file.open('a') as f:
        f.write(f"{video_path.name}\n")
        
def extract_every_nth_frame_from_videos_in_folder(
    input_folder: Path,
    output_root_folder: Path,
    frame_step: int,
    error_log_file: Path,
    processed_videos_file: Path,
):
    """
    Extract frames from videos, skipping existing frames.

    Args:
        input_folder (Path): Path to the folder containing video files.
        output_root_folder (Path): Path to the root folder where frames are saved.
        frame_step (int): Interval of frames to extract (e.g., every 10th frame).
        error_log_file (Path): Path to the text file where errors will be logged.
        processed_videos_file (Path): Path to file tracking processed videos.
    """
    # Ensure output root folder exists
    output_root_folder.mkdir(parents=True, exist_ok=True)

    # Get all MP4 files in the input folder
    video_files = list(input_folder.glob("*.MP4"))
    if not video_files:
        logging.info(f"No .MP4 files found in the folder: {input_folder}")
        return
    
    # Get list of already processed videos
    processed_videos = get_processed_videos(processed_videos_file)
    videos_to_process = [v for v in video_files if v.name not in processed_videos]
    
    logging.info(f"Found {len(video_files)} total videos")
    logging.info(f"Already processed: {len(processed_videos)} videos")
    logging.info(f"Remaining to process: {len(videos_to_process)} videos")
    
    for video_path in video_files:
        # Create output folder for this video
        video_output_folder = output_root_folder / video_path.stem
        video_output_folder.mkdir(parents=True, exist_ok=True)

        # Extract frames
        logging.info(f"Processing video: {video_path}")
        extract_every_nth_frame(
            video_path=video_path,
            output_folder=video_output_folder,
            frame_interval=frame_step,
            error_log_file=error_log_file
        )

        mark_video_as_processed(video_path, processed_videos_file)

    logging.info(f"All videos processed. Frames saved to {output_root_folder}")
    
def extract_every_nth_frame(video_path: Path, output_folder: Path, frame_interval: int, error_log_file: Path):
    """
    Extract every nth frame from a video, ensuring exact frame positioning.

    Args:
        video_path (Path): Path to the input video file.
        output_folder (Path): Path to the folder where frames will be saved.
        frame_interval (int): Interval of frames to extract (e.g., every 10th frame).
        error_log_file (Path): Path to the text file where errors will be logged.
    """
    # Ensure the output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"Error: Unable to open video file {video_path}")
        with Path(error_log_file).open('a') as error_log:
            error_log.write(f"Failed to open video: {video_path}\n")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    failed_frames = []

    def generate_image_name(video_path: Path, frame_idx: int) -> str:
        return f"{video_path.stem}_{frame_idx:06d}.jpg"

    saved_frames = 0
    with tqdm(total=total_frames // frame_interval, desc="Extracting frames") as pbar:
        frame_idx = 0
        while frame_idx < total_frames:
            frame_name = generate_image_name(video_path, frame_idx)
            output_path = output_folder / frame_name

            # Set frame position explicitly before reading
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                logging.error(f"Error: Unable to read frame {frame_idx} in {video_path}")
                failed_frames.append((video_path.name, frame_idx))
                frame_idx += frame_interval
                pbar.update(1)
                continue

            try:
                # Try to save the extracted frame
                cv2.imwrite(str(output_path), frame)
                saved_frames += 1
            except Exception as e:
                logging.error(f"Error saving frame {frame_idx} from {video_path}: {str(e)}")
                failed_frames.append((video_path.name, frame_idx))

            frame_idx += frame_interval
            pbar.update(1)

    cap.release()

    # Write all failed frames to error log at once
    if failed_frames:
        with Path(error_log_file).open('a') as error_log:
            for video_name, frame_idx in failed_frames:
                error_log.write(f"{video_name}, frame {frame_idx}\n")

    logging.info(f"Extraction complete for {video_path}:")
    logging.info(f"Saved {saved_frames} frames")
    logging.info(f"Failed to extract {len(failed_frames)} frames")


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

def extract_audio_from_video(video_path: Path, audio_output_path: Path) -> None:
    """
    Extracts the audio from a video file and saves it directly as a 16kHz WAV file.
    
    Parameters
    ----------
    video_path : Path
        Path to the input video file.
    audio_output_path : Path
        Path where the 16kHz WAV file should be saved.
    """
    parent_dir = audio_output_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Use ffmpeg directly to extract and convert audio in one step
        process = subprocess.run([
            'ffmpeg',
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output
            str(audio_output_path)
        ], check=True, capture_output=True, text=True)

        if process.returncode == 0:
            logging.info(f"Successfully stored 16kHz audio at {audio_output_path}")
        else:
            logging.error(f"Error extracting audio: {process.stderr}")

    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error: {e.stderr}")
    except Exception as e:
        logging.error(f"Error extracting audio: {e}")

def extract_audio_from_videos_in_folder(videos_input_dir: Path, output_dir: Path):
    """
    Extracts audio from all video files in the specified folder, if not already done.
    """
    logging.info(f"Scanning folder: {videos_input_dir}")  # Debugging step

    # load file with problematic videos
    problematic_videos = []
    with open("/home/nele_pauline_suffo/outputs/audio_extraction/problematic_audio_files.txt", 'r') as f:
        problematic_videos = [line.strip() for line in f.readlines()]
    for video_file in videos_input_dir.iterdir():
        
        if video_file.suffix.lower() not in ['.mp4', '.MP4'] or video_file.name in problematic_videos:
            logging.info(f"Skipping problematic video file: {video_file}")
            continue
        
        # skip if video file is already in the output directory
        if (output_dir / f"{video_file.stem}.wav").exists():
            logging.info(f"Audio already exists for: {video_file}")
            continue
        
        # create output directory if it doesn't exist
        audio_output_path = output_dir / f"{video_file.stem}.wav"

        if not audio_output_path.exists():
            logging.info(f"Extracting audio for: {video_file}")  # Debugging step
            extract_audio_from_video(video_file, audio_output_path)
            logging.info(f"Finished processing: {video_file}")  # Debugging step
        else:
            logging.info(f"Audio already exists for: {video_file}")  # Debugging step
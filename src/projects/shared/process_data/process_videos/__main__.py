import logging
from pathlib import Path
import concurrent.futures
from src.projects.shared.process_data.process_videos.utils import extract_frames_from_single_video
from src.projects.social_interactions.common.constants import DetectionParameters, VideoParameters, DetectionPaths, YoloParameters

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_video(
    video_file: Path, 
    output_dir: Path, 
    fps: int
) -> tuple:
    """
    This function processes a single video file by extracting frames from it.
    Returns a tuple with the success status and the video file.
    
    Parameters
    ----------
    video_file : Path
        the video file to process
    output_dir : Path
        the directory to save the extracted frames
    fps : int
        the frames per second to extract
    """
    try:
        success = extract_frames_from_single_video(video_file, output_dir, fps)
        return success, video_file

    except Exception as e:
        logging.error(f"Error processing video {video_file}: {e}")
        return False, video_file


def extract_frames_from_all_videos(
    video_dir: str, 
    output_dir: str, 
    fps: int,
) -> None:
    """
    This function extracts frames from all video files in the given directory and saves them as images in the output directory.

    Parameters
    ----------
    video_dir : str
        The directory containing the video files.
    output_dir : str
        The directory to save the extracted frames.
    fps : int
        The frames per second to extract.
    """
    logging.info(f"Starting frame extraction from videos in {video_dir} to {output_dir} at {fps} FPS.")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get a list of all video files in the directory
    video_files = list(Path(video_dir).glob('*' + DetectionParameters.file_extension.lower()))
    video_files += list(Path(video_dir).glob('*' + DetectionParameters.file_extension.upper()))
    logging.info(f"Found {len(video_files)} video files to process.")
    
    # List to store successfully processed videos
    successful_videos = []  

    # Dictionary to keep track of retries
    batch_size = VideoParameters.batch_size
    
    def process_batch(batch: list):
        """
        This function processes a batch of video files concurrently.

        Parameters
        ----------
        batch : list
            the list of video files to process
        """         
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for video_file in batch:
                future = executor.submit(process_video, video_file, output_dir, fps)
            for completed_future in concurrent.futures.as_completed([future]):
                success, video_file = completed_future.result()
                if success:
                    # Store the name of the successful video
                    successful_videos.append(video_file.name)  
    
    for i in range(0, len(video_files), batch_size):
        batch = video_files[i:i + batch_size]
        logging.info(f"Processing batch {i // batch_size + 1} with {len(batch)} videos.")
        process_batch(batch)
        logging.info(f"Completed batch {i // batch_size + 1}.")


    logging.info("Completed frame extraction for all videos.")
    if successful_videos:
        # Write the list of successful videos to a log file
        success_log_path = VideoParameters.success_log_path
        # Create the directory if it does not exist
        success_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with success_log_path.open("w") as file:
            for video_name in successful_videos:
                file.write(f"{video_name}\n")
        logging.info(f"Successfully processed videos are listed in {success_log_path}")


def main() -> None:
    output_dir = DetectionPaths.images_input
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    # Extract frames from video
    extract_frames_from_all_videos(DetectionPaths.videos_input, output_dir, YoloParameters.fps)

if __name__ == "__main__":
    main()

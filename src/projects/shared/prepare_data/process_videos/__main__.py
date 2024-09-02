import logging
from pathlib import Path
import concurrent.futures
from src.projects.shared.prepare_data.process_videos.utils import extract_frames_from_single_video
from src.projects.social_interactions.common.constants import DetectionParameters, VideoParameters, DetectionPaths, YoloParameters

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

batch_size = VideoParameters.batch_size


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

    # Ensure output directory and success log path exist
    output_dir.mkdir(parents=True, exist_ok=True)
    VideoParameters.success_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get a list of all video files in the directory
    video_files = list(Path(video_dir).glob('*' + DetectionParameters.file_extension.lower()))
    video_files += list(Path(video_dir).glob('*' + DetectionParameters.file_extension.upper()))
    logging.info(f"Found {len(video_files)} video files to process.")
    
    
    def process_batch(batch: list):
        """
        This function processes a batch of video files concurrently.

        Parameters
        ----------
        batch : list
            the list of video files to process
        """         
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Process each video in the batch concurrently
            futures = [executor.submit(process_video, video_file, output_dir, fps) for video_file in batch]
            for completed_future in concurrent.futures.as_completed(futures):
                success, video_file = completed_future.result()  
    
    for i in range(0, len(video_files), batch_size):
        batch = video_files[i:i + batch_size]
        logging.info(f"Processing batch {i // batch_size + 1} with {len(batch)} videos.")
        process_batch(batch)
        logging.info(f"Completed batch {i // batch_size + 1}.")


    logging.info("Completed frame extraction for all videos.")
    logging.info(f"Successfully processed videos are listed in {VideoParameters.success_log_path}")


def main() -> None:
    output_dir = DetectionPaths.images_input
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    # Extract frames from video
    extract_frames_from_all_videos(DetectionPaths.videos_input, output_dir, YoloParameters.fps)

if __name__ == "__main__":
    main()

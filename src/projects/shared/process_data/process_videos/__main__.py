import logging
from pathlib import Path
import concurrent.futures
from src.projects.shared.process_data.process_videos.utils import extract_frames_from_single_video
from src.projects.social_interactions.common.constants import DetectionPaths, DetectionParameters, YoloParameters

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_frames_from_all_videos(
    video_dir: str, 
    output_dir: str, 
    fps: int,
) -> None:
    """
    This function extracts frames from a video file and saves them as images in the output directory.

    Parameters
    ----------
    video_dir : str
        the directory containing the video files
    output_dir : str
        the directory to save the extracted frames
    fps : int
        the frames per second to extract
    """
    logging.info(f"Starting frame extraction from videos in {video_dir} to {output_dir} at {fps} FPS.")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get a list of all video files in the directory
    video_files = list(Path(video_dir).glob('*' + DetectionParameters.file_extension.lower()))
    video_files += list(Path(video_dir).glob('*' + DetectionParameters.file_extension.upper()))
    logging.info(f"Found {len(video_files)} video files to process.")

    # Use a ProcessPoolExecutor to process videos in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for video_file in video_files:
            executor.submit(extract_frames_from_single_video, video_file, output_dir, fps)

    logging.info("Completed frame extraction for all videos.")

def main() -> None:
    output_dir = DetectionPaths.images_input
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    # Extract frames from video
    extract_frames_from_all_videos(DetectionPaths.videos_input, output_dir, YoloParameters.fps)

if __name__ == "__main__":
    main()

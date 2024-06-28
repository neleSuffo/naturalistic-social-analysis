from pathlib import Path
import concurrent.futures
from shared.process_data.process_videos.utils import extract_frames_from_single_video
from projects.social_interactions.src.common.constants import DetectionPaths, DetectionParameters, YoloParameters



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
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get a list of all video files in the directory
    video_files = list(Path(video_dir).glob('*' + DetectionParameters.file_extension.lower()))
    video_files += list(Path(video_dir).glob('*' + DetectionParameters.file_extension.upper()))

    # Use a ProcessPoolExecutor to process videos in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for video_file in video_files:
            executor.submit(extract_frames_from_single_video, video_file, output_dir, fps)



def main() -> None:
    output_dir = DetectionPaths.images_input
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    # Extract frames from video
    extract_frames_from_all_videos(DetectionPaths.videos_input, output_dir, YoloParameters.fps)

if __name__ == "__main__":
    main()
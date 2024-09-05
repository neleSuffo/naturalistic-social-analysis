import logging
from src.projects.shared.utils import extract_frames_from_all_videos 
from src.projects.social_interactions.common.constants import DetectionPaths, YoloParameters, ModelsPreprocessing as MP

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main() -> None:
    output_dir = DetectionPaths.images_input
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    # Extract frames from video
    extract_frames_from_all_videos(DetectionPaths.videos_input, output_dir, YoloParameters.fps, MP.yolo)

if __name__ == "__main__":
    main()

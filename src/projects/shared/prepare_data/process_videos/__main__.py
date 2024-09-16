import logging
from src.projects.shared.utils import extract_frames_from_videos 
from src.projects.social_interactions.common.constants import DetectionPaths, ModelNames as MN
from src.projects.social_interactions.config.config import YoloConfig as YC

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main() -> None:
    output_dir = DetectionPaths.images_input_dir
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    # Extract frames from video
    extract_frames_from_videos(DetectionPaths.videos_input_dir, output_dir, YC.extraction_fps, MN.yolo_model)

if __name__ == "__main__":
    main()

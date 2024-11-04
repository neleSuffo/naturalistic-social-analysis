import logging
from utils import extract_frames_from_videos 
from constants import DetectionPaths, ModelNames
from config import YoloConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main() -> None:
    output_dir = DetectionPaths.images_input_dir
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    # Extract frames from video
    extract_frames_from_videos(DetectionPaths.videos_input_dir, output_dir, YoloConfig.extraction_fps, ModelNames.yolo_model)

if __name__ == "__main__":
    main()

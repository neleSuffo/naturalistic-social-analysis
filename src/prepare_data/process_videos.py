import logging
from utils import extract_every_nth_frame_from_videos_in_folder 
from constants import DetectionPaths, VideoParameters, VTCPaths
from config import DetectionParameters
from utils import extract_audio_from_videos_in_folder
from prepare_data.crop_detections import main as crop_detections

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main() -> None: 
    #logging.info("Starting to extract frames from videos.")
    # Extract frames from video
    #extract_every_nth_frame_from_videos_in_folder(DetectionPaths.quantex_videos_input_dir, DetectionPaths.images_input_dir, DetectionParameters.frame_step_interval, VideoParameters.rawframes_extraction_error_log)
    #extract_every_nth_frame_from_videos_in_folder(DetectionPaths.childlens_videos_input_dir, DetectionPaths.childlens_images_input_dir, DetectionParameters.frame_step_interval, VideoParameters.rawframes_extraction_error_log)
    #logging.info("Finished extracting frames from videos.")
    
    logging.info("Starting to crop detections from labels.")
    # extract persons and faces from the images
    #crop_detections()
    logging.info("Finished cropping detections from labels.")
    
    logging.info("Starting to extract audio from videos.")
    #Extract audio from video
    extract_audio_from_videos_in_folder(DetectionPaths.quantex_videos_input_dir, VTCPaths.quantex_audio_dir)
    logging.info("Finished extracting audio from videos.")
    

    
if __name__ == "__main__":
    main()

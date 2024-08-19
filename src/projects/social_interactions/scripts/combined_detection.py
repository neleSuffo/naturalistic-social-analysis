import os
import json
import logging
from multiprocessing.pool import ThreadPool
from facenet_pytorch import MTCNN
from pathlib import Path
from timeit import default_timer as timer
from src.projects.social_interactions.models.mtcnn.run_mtcnn import run_mtcnn
from src.projects.social_interactions.models.yolo_inference.run_yolo import run_yolo
from src.projects.social_interactions.scripts.language import detect_voices
from src.projects.social_interactions.common.constants import (
    DetectionPaths,
    DetectionParameters,
)
from multiprocessing import Value
from src.projects.social_interactions.common import my_utils
from typing import Dict
from threading import Lock

 
 # Set up logging
logging.basicConfig(level=logging.INFO)   

class SharedResources:
    def __init__(self):
        #  Initialize a lock for thread-safe operations
        self.lock = Lock()
        # Initialize shared annotation_id and image_id
        # This variable is used to assign unique IDs to the annotations when running the detection models in parallel
        self.annotation_id = Value("i", 1)
        self.image_id = Value("i", 1)
        self.video_file_ids_dict = {}  
        
    def create_video_id_mapping(self):
        return my_utils.create_video_to_id_mapping(
            [video_file.stem for video_file in DetectionPaths.videos_input.iterdir()
             if video_file.suffix.lower() == DetectionParameters.file_extension]
        )


class OutputMerger:
    def __init__(self):
        self.combined_output = {
            DetectionParameters.output_key_videos: [],
            DetectionParameters.output_key_annotations: [],
            DetectionParameters.output_key_images: [],
            DetectionParameters.output_key_categories: [],
        }

    def merge(self, detection_output: dict):
        # Get existing video IDs to avoid duplicates
        existing_video_ids = {video["id"] for video in self.combined_output[DetectionParameters.output_key_videos]}
        
        # Add new videos to the combined COCO output
        for video in detection_output.get(DetectionParameters.output_key_videos, []):
            if video["id"] not in existing_video_ids:
                self.combined_output[DetectionParameters.output_key_videos].append(video)
        
        # Add annotations and images
        self.combined_output[DetectionParameters.output_key_annotations].extend(detection_output.get(DetectionParameters.output_key_annotations, []))
        self.combined_output[DetectionParameters.output_key_images].extend(detection_output.get(DetectionParameters.output_key_images, []))
        self.combined_output[DetectionParameters.output_key_categories].extend(detection_output.get(DetectionParameters.output_key_categories, []))
    
    def get_combined_output(self):
        return self.combined_output
    

class Detector:
    def __init__(self, shared_resources: SharedResources, output_merger: OutputMerger):
        self.shared_resources = shared_resources
        self.output_merger = output_merger
        
        # Initialize processors
        self.yolo_processor = YOLOProcessor(output_merger, shared_resources)
        self.mtcnn_processor = MTCNNProcessor(output_merger, shared_resources)
        self.voice_processor = VoiceTypeProcessor(output_merger, shared_resources)
        
    def process_video_file(self, video_file: Path, detections: Dict[str, bool]) -> None:
        """
        This function processes a video file by performing the specified detections.
        
        Parameters
        ----------
        video_file : Path
            the path to the video file to process
        detections : dict
            the detections to perform
        """
        # Get the video file name
        file_name_short = video_file.stem
        logging.info(f"Processing {file_name_short}...")

        # Add the video file to the output
        self.output_merger.combined_output[DetectionParameters.output_key_videos].append(
            {
                "id": self.shared_resources.video_file_ids_dict[file_name_short], 
                "file_name": video_file.name,
            }
        )

        # Run the desired detections
        if detections.get("person", False):
            self.yolo_processor.run_person_detection(video_file)
        if detections.get("face", False):
            self.mtcnn_processor.run_face_detection(video_file)
        if detections.get("voice", False):
            self.voice_processor.process_all()
            
    def wrapper(self, args: tuple) -> None:
        """
        This function is a wrapper for the process_video_file function.

        Parameters
        ----------
        args : tuple
            the arguments for the process_video_file function

        """
        video_file, detections = args
        return self.process_video_file(video_file, detections)

    def run_detection(self, detections: dict, batch_size) -> dict:
        """
        This function runs the detection models in batches.
        
        Parameters
        ----------
        detections : dict
            the detections to perform
        batch_size : int
            the batch size for processing videos
        """
        # Get a list of all video files in the folder
        video_files = [
            video_f 
            for video_f in DetectionPaths.videos_input.iterdir() 
            if video_f.suffix.lower() == DetectionParameters.file_extension
        ]
        # Process the videos in batches
        try:
            total_batches = (len(video_files) + batch_size - 1) // batch_size
            for i in range(0, len(video_files), batch_size):
                batch = video_files[i:i + batch_size]
                batch_number = (i // batch_size) + 1
                logging.info(f"Processing batch {batch_number}/{total_batches} with {len(batch)} videos...")
                # Process the videos in parallel
                with ThreadPool() as pool:
                    pool.map(
                        self.wrapper,
                        [
                            (video_file, detections) 
                            for video_file in batch
                        ],
                    )
        except Exception as e:
            logging.error(f"An error occurred during detection: {e}")
            raise
        
        return self.output_merger.get_combined_output()           
            
            
class MTCNNProcessor:
    def __init__(self, output_merger: OutputMerger, shared_resources):
        self.output_merger = output_merger
        # Use the shared resources
        self.video_file_ids_dict = shared_resources.video_file_ids_dict
        self.annotation_id = shared_resources.annotation_id
        self.image_id = shared_resources.image_id
        # Pass the MTCNN model instance
        # keep_all=True to get all faces in the image
        self.model = MTCNN(keep_all=True, device='cuda:0')

        # Initialize the list of existing image file names
        self.existing_image_file_names_with_ids = {}

    def run_face_detection(self, video_file):
        logging.info("Running face detection...")
        detection_output = run_mtcnn(
            video_file, 
            self.video_file_ids_dict, 
            self.annotation_id, 
            self.image_id, 
            self.model,
            self.existing_image_file_names_with_ids,
        )
        
        if detection_output:
            # Merge the results into the combined output
            self.output_merger.merge(detection_output)


class YOLOProcessor:
    def __init__(self, output_merger: OutputMerger, shared_resources):
        self.output_merger = output_merger
        # Use the video_file_ids_dict from the Detector class
        self.video_file_ids_dict = shared_resources.video_file_ids_dict
        # Use the shared annotation_id and image_id
        self.annotation_id = shared_resources.annotation_id
        self.image_id = shared_resources.image_id

    def run_person_detection(self, video_file):
        logging.info("Running person detection...")
        detection_output = run_yolo(
            video_file, 
            self.video_file_ids_dict, 
            self.annotation_id, 
            self.image_id, 
            self.model,
            self.existing_image_file_names_with_ids,
        )
        
        if detection_output:
            # Merge the results into the combined output
            self.output_merger.merge(detection_output)
    
class VoiceTypeProcessor:
    def __init__(self, output_merger: OutputMerger, shared_resources):
        self.output_merger = output_merger
        # Use the video_file_ids_dict from the Detector class
        self.video_file_ids_dict = shared_resources.video_file_ids_dict
        # Use the shared annotation_id and image_id
        self.annotation_id = shared_resources.annotation_id
        self.image_id = shared_resources.image_id
    
    def extract_audio_from_videos(self):
        """This function extracts audio from videos for voice detection."""
        logging.info("Extracting audio for voice detection...")
        my_utils.extract_audio_from_videos_in_folder(DetectionPaths.videos_input)
        
    def run_voice_detection(self):
        """This function runs voice detection on the extracted audio files.

        Returns
        -------
        dict
            the detection results in COCO format
        """
        logging.info("Running voice detection...")
        detection_output = detect_voices.run_voice_detection(self.video_file_ids_dict, self.annotation_id, self.image_id)

        # Merge the results into the combined output
        self.output_merger.merge(detection_output)
        
    def process_all(self):
        """This function extracts audio from videos and runs voice detection."""
        self.extract_audio_from_videos()
        detection_output = self.run_voice_detection()
        return detection_output 


def main(detections: dict, batch_size: int) -> None:
    """
    This function runs the social interactions detection pipeline.

    Parameters
    ----------
    detections : dict
        the detections to perform
    batch_size : int
        the batch size for processing videos
    """
    logging.info("Starting detection process...")
    start_time = timer()
    output_merger = OutputMerger()
    shared_resources = SharedResources()
        
    # Create video file IDs if they don't already exist
    if not shared_resources.video_file_ids_dict:
        video_file_mapping = shared_resources.create_video_id_mapping()
        logging.info(f"Video file IDs mapping created")
        shared_resources.video_file_ids_dict = video_file_mapping

    # Instantiate the detector and run the detection process
    detector = Detector(shared_resources, output_merger)
    combined_output = detector.run_detection(detections, batch_size)
    
    # Write the combined output to a JSON file
    with DetectionPaths.combined_json_output_path.open('w') as file:
        json.dump(combined_output, file, indent=4)    
    
    end_time = timer()
    runtime = end_time - start_time
    logging.info(f"Detection process completed. Runtime: {runtime} seconds")


if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    # Define which detections to perform
    detections = {"person": False, "face": True, "voice": False}
    batch_size = 2
    main(detections, batch_size)


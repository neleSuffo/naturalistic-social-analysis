import os
import json
import psutil
import logging
from multiprocessing.pool import ThreadPool
from pathlib import Path
from timeit import default_timer as timer
from src.projects.social_interactions.models.mtcnn import run_mtcnn
from src.projects.social_interactions.models.yolo_inference import run_yolov5
from src.projects.social_interactions.scripts.language import detect_voices
from src.projects.social_interactions.common.constants import (
    DetectionPaths,
    LabelToCategoryMapping,
    DetectionParameters,
)
from multiprocessing import Value
from src.projects.social_interactions.common import my_utils
from src.projects.shared import utils as shared_utils
from typing import Dict, Callable
from threading import Lock

# Set up logging
logging.basicConfig(level=logging.INFO)

class Detector:
    def __init__(self):
        #  Initialize a lock for thread-safe operations
        self.lock = Lock()

        # Define the detection functions
        self.detection_functions = {
            "person": run_yolov5.run_person_detection,
            "face": run_mtcnn.run_face_detection,
            "voice": detect_voices.run_voice_detection,
        }

        # Define the load model functions
        self.models = {
            "person": shared_utils.load_yolov5_model,
            "face": shared_utils.load_mtcnn_model,
        }

        # Create a dictionary mapping video file names to IDs
        self.video_file_ids_dict = my_utils.create_video_to_id_mapping(
            [
                video_file.stem
                for video_file in Path(DetectionPaths.videos_input).iterdir()
                if video_file.suffix.lower() == DetectionParameters.file_extension
            ]
        )

        # Load existing annotation data to find the highest annotation ID
        if DetectionPaths.combined_json_output_path.exists():
            with DetectionPaths.combined_json_output_path.open('r') as file:
                existing_data = json.load(file)
                # Check for existing images and annotations
                existing_image_ids = [image['id'] for image in existing_data.get('images', [])]
                self.existing_image_file_names = [image['file_name'] for image in existing_data.get('images', [])]
                existing_annotation_ids = [annotation['id'] for annotation in existing_data.get('annotations', [])]
                max_existing_annotation_id = max(existing_annotation_ids, default=0)
                max_existing_image_id = max(existing_image_ids, default=0)
        else:
            max_existing_annotation_id = 0
            max_existing_image_id = 0
            self.existing_image_file_names = []

        # Initialize shared annotation_id and image_id
        # This variable is used to assign unique IDs to the annotations when running the detection models in parallel
        self.annotation_id = Value("i", max_existing_annotation_id + 1)
        self.image_id = Value("i", max_existing_image_id + 1)


    def call_models(
        self,
        detection_type: str,
        video_file: Path,
        file_name_short: str,
        models: Dict[str, Callable],
        output: dict,
    ) -> dict:
        """
        This function performs a specific detection type on a video file.

        Parameters
        ----------
        detection_type : str
            the type of detection to perform
        video_file : path
            the path to the video file to process
        file_name_short : str
            the name of the video file without the extension
        models : dict
            the models to use for detection
        output : dict
            the output dictionary

        Returns
        -------
        dict
            the updated output dictionary
        """
        logging.info(f"Performing {detection_type} detection...")
        detection_function = self.detection_functions[detection_type]
        # Perform the detection
        # Get the category ID and supercategory from the dictionaries
        category_id = LabelToCategoryMapping.label_dict[detection_type]
        supercategory = LabelToCategoryMapping.supercategory_dict[category_id]
        # Create category dictionary and append it to output["categories"]
        category = {
            "id": category_id,
            "name": detection_type,
            "supercategory": supercategory,
        }
        if category not in output["categories"]:
            output["categories"].append(category)

        # Get the unique video file ID from the video file name
        file_id = self.video_file_ids_dict[file_name_short]
        # Define dictionary that maps detection types to their corresponding arguments
        args = {
            "video_input_path": video_file,                        
            "annotation_id": self.annotation_id,
            "image_id": self.image_id,
            "video_file_name": file_name_short,
            "file_id": file_id,
            "model": models.get(detection_type),
            "existing_image_file_names": self.existing_image_file_names,
        }

        # Call the function with the appropriate arguments
        detections = detection_function(**args)
        
        # Add detection_output to "images" and "annotations"
        output["images"].extend(detections["images"])
        output["annotations"].extend(detections["annotations"])

        return output

    def process_video_file(
        self,
        video_file: Path,
        detections: Dict[str, bool],
        models: Dict[str, Callable],
        combined_coco_output: dict,
    ) -> None:
        """
        This function processes a video file by performing the specified detections.

        Parameters
        ----------
        video_file : Path
            the path to the video file to process
        detections : dict
            the detections to perform
        models : dict
            the models to use for detection
        combined_coco_output : dict
            the combined output dictionary for all videos
        """
        # Initialize output dictionary for the current video file
        output = {
            "images": [],
            "annotations": [],
            "categories": [],
        }
        # Get the video file name
        file_name = video_file.name
        file_name_short = video_file.stem
        logging.info(
            f"Starting social interactions detection pipeline for {file_name_short}..."
        )    
        
        # Add the video file to the output
        combined_coco_output["videos"].append(
            {
                "id": self.video_file_ids_dict[file_name_short],
                "file_name": file_name,
            }
        )
        # Perform the desired detections
        for detection_type in self.detection_functions.keys():
            if detections[detection_type] and detection_type != "voice":
                output = self.call_models(
                    detection_type,
                    video_file,
                    file_name_short,
                    models,
                    combined_coco_output,
                )

        # Ensure thread-safe update
        with self.lock:  
            for category in output["categories"]:
                if category not in combined_coco_output["categories"]:
                    combined_coco_output["categories"].append(category)
    

    def wrapper(self, args: tuple) -> None:
        """
        This function is a wrapper for the process_video_file function.

        Parameters
        ----------
        args : tuple
            the arguments for the process_video_file function
        """
        video_file, detections, models_dict, combined_coco_output = args
        return self.process_video_file(
            video_file, detections=detections, models=models_dict, combined_coco_output=combined_coco_output
        )

    def run_detection(
        self,
        detections: dict,
        batch_size: int = 10,
    ) -> dict:
        """
        This function runs four different detection models:
        - Person detection
        - Face detection
        - Voice detection

        Parameters
        ----------
        detections : dict
            a dictionary indicating which detection models to run
            (person, face, voice)
        batch_size : int
            the batch size for processing video files in parallel

        Returns
        -------
        dict
            the results of each detection model
        file_name_short
            the name of the video file (without the extension)
        """
        # Load the desired models
        models_dict = {
            key: self.models[key]()
            for key, value in detections.items()
            if value and key in self.models
        }

        # Get a list of all video files in the folder
        video_files = [
            video_f
            for video_f in Path(DetectionPaths.videos_input).iterdir()
            if video_f.suffix.lower() == DetectionParameters.file_extension
        ]

        # Initialize the combined COCO output dictionary
        combined_coco_output = {
            "videos": [],
            "images": [],
            "annotations": [],
            "categories": [],
        }
        
        if detections.get("voice", False):
            # Extract audio from videos
            logging.info("Extracting audio for voice detection...")
            my_utils.extract_audio_from_videos_in_folder(DetectionPaths.videos_input)
        try:
            total_batches = (len(video_files) + batch_size - 1) // batch_size
            for i in range(0, len(video_files), batch_size):
                batch = video_files[i:i + batch_size]
                batch_number = (i // batch_size) + 1
                logging.info(f"Processing batch {batch_number}/{total_batches} with {len(batch)} videos...")
                with ThreadPool() as pool:
                    pool.map(
                        self.wrapper,
                        [
                            (video_file, detections, models_dict, combined_coco_output) 
                            for video_file in batch
                        ],
                    )
        except Exception as e:
            logging.error(f"An error occurred during detection: {e}")
            raise
        
        # Run voice detection if enabled, only once
        if detections.get("voice", False):
            logging.info("Running voice detection once across all videos...")
            detection_output = detect_voices.run_voice_detection(
                self.video_file_ids_dict, # the dictionary mapping video file names to IDs
                self.annotation_id,
                self.image_id
            )
            
            # Merge voice detection results into the combined COCO output
            existing_video_ids = {video["id"] for video in combined_coco_output["videos"]}
            # Add new videos to the combined COCO output
            for video in detection_output["videos"]:
                if video["id"] not in existing_video_ids:
                    combined_coco_output["videos"].append(video)
            combined_coco_output["annotations"].extend(detection_output["annotations"])
            combined_coco_output["images"].extend(detection_output["images"])
            
        # Save the results to a JSON file
        # Check if a results file already exists
        #my_utils.update_or_create_json_file(DetectionPaths.combined_json_output_path, combined_coco_output)
        #for now:
            # Write the COCO output directly to the file
        with DetectionPaths.combined_json_output_path.open('w') as file:
            json.dump(combined_coco_output, file, indent=4)


def main(detections_dict: dict) -> None:
    """
    The main function of the social interactions detection pipeline

    Parameters
    ----------
    detections_dict : dict
        a dictionary indicating which detection models to run
        (person, face, voice, proximity)

    """
    logging.info("Starting detection process...")
    start_time = timer()
    detector = Detector()
    detector.run_detection(detections_dict)
    end_time = timer()
    runtime = end_time - start_time
    logging.info(f"Detection process completed. Runtime: {runtime} seconds")


if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '2'
    logging.basicConfig(level=logging.INFO)
    detections_dict = {
        "person": False,
        "face": True,
        "voice": False,
    }
    main(detections_dict)

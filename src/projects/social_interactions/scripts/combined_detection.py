from multiprocessing.pool import ThreadPool
from pathlib import Path
from timeit import default_timer as timer
from projects.social_interactions.models.mtcnn import run_mtcnn
from projects.social_interactions.models.yolov5_inference import run_yolov5
from projects.social_interactions.scripts.language import detect_voices
from projects.social_interactions.common.constants import (
    DetectionPaths,
    LabelToCategoryMapping,
    DetectionParameters,
)
from multiprocessing import Value
from projects.social_interactions.common import my_utils
from projects.shared import utils as shared_utils
from typing import Dict, Callable
import logging
import copy

# Set up logging
logging.basicConfig(level=logging.INFO)

class Detector:
    def __init__(self):
        # Define the detection functions
        self.detection_functions = {
            "person": run_yolov5.run_person_detection,
            "face": run_mtcnn.run_face_detection,
            "voice": detect_voices.run_voice_detection,
            "proximity": my_utils.run_proximity_detection,
        }

        # Define the load model functions
        self.models = {
            "person": shared_utils.load_yolov5_model,
            "face": shared_utils.load_mtcnn_model,
        }

        # Create a dictionary mapping video file names to IDs
        self.video_file_ids = my_utils.create_video_to_id_mapping(
            [
                video_file.name
                for video_file in Path(DetectionPaths.videos_input).iterdir()
                if video_file.suffix.lower() == DetectionParameters.file_extension
            ]
        )
        # Create a shared annotation_id variable
        self.annotation_id = Value("i", 0)

    def call_models(
        self,
        detection_type: str,
        video_file: Path,
        file_name: str,
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
        file_name : str
            the name of the video file
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
        if detection_type in ["person", "face", "voice"]:
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

            file_id = self.video_file_ids[file_name]
            # Define dictionary that maps detection types to their corresponding arguments
            args = {
                "video_input_path": video_file,
                "annotation_id": self.annotation_id,
            }

            # Conditionally add additional arguments for "face" and "person" detection types
            if detection_type in ["person", "face"]:
                args.update(
                    {
                        "video_file_name": file_name,
                        "file_id": file_id,
                        "model": models.get(detection_type),
                    }
                )

            # Call the function with the appropriate arguments
            detections = detection_function(**args)
            # Add detection_output to "images" and "annotations"
            output["images"].extend(detections["images"])
            output["annotations"].extend(detections["annotations"])

        elif detection_type == "proximity":
            logging.info("Performing proximity detection...")
            detection_function()

        return output

    def process_video_file(
        self,
        video_file: Path,
        detections: Dict[str, bool],
        models: Dict[str, Callable],
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
        """
        file_name = video_file.name
        file_name_short = video_file.stem
        logging.info(
            f"Starting social interactions detection pipeline for {file_name_short}..."
        )

        # Initialize the COCO structure template
        coco_template = {
            "videos": [],
            "images": [],
            "annotations": [],
            "categories": [],
        }

        # Use deepcopy to create a new instance of coco_template
        output = copy.deepcopy(coco_template)

        # Add the video file to the output
        output["videos"].append(
            {
                "id": self.video_file_ids[file_name],
                "file_name": f"{file_name_short}{DetectionParameters.file_extension}",
            }
        )
        # Perform the desired detections
        for detection_type in self.detection_functions.keys():
            if detections[detection_type]:
                output = self.call_models(
                    detection_type,
                    video_file,
                    file_name,
                    models,
                    output,
                )
        # Save the result to a JSON file
        json_output_path = DetectionPaths.results / f"{file_name_short}_detections.json"
        my_utils.save_results_to_json(output, json_output_path)


    def wrapper(self, args: tuple) -> None:
        """
        This function is a wrapper for the process_video_file function.

        Parameters
        ----------
        args : tuple
            the arguments for the process_video_file function
        """
        video_file, detections, models_dict = args
        return self.process_video_file(
            video_file, detections=detections, models=models_dict
        )

    def run_detection(
        self,
        detections: dict,
    ) -> dict:
        """
        This function runs five different detection models:
        - Person detection
        - Face detection
        - Voice detection
        - Proximity detection

        Parameters
        ----------
        detections : dict
            a dictionary indicating which detection models to run
            (person, face, voice, proximity)

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

        try:
            # Process each video file (in parallel)
            with ThreadPool() as pool:
                pool.map(
                    self.wrapper,
                    [
                        (video_file, detections, models_dict)
                        for video_file in video_files
                    ],
                )
        except Exception as e:
            logging.error(f"An error occurred during detection: {e}")
            raise


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
    logging.basicConfig(level=logging.INFO)
    detections_dict = {
        "person": True,
        "face": True,
        "voice": False,
        "proximity": False,
    }
    main(detections_dict)
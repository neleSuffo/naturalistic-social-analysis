from multiprocessing.pool import ThreadPool
from pathlib import Path
from timeit import default_timer as timer
from src.social_interaction.persons import detect_persons
from faces import detect_faces
from language import detect_voices
from constants import DetectionPaths, LabelDictionaries
from multiprocessing import Value
import my_utils
import logging
import copy
import os


# Set up logging
logging.basicConfig(level=logging.INFO)


class Detector:
    def __init__(self):
        # Define the detection functions
        self.detection_functions = {
            "person": detect_persons.run_person_detection,
            "face": detect_faces.run_face_detection,
            "voice": detect_voices.run_voice_detection,
            "proximity": my_utils.run_proximity_detection,
        }

        # Define the load model functions
        self.models = {
            "person": my_utils.load_person_detection_model,
            "face": my_utils.load_frame_face_detection_model,
        }

        # Create a dictionary mapping video file names to IDs
        self.video_file_ids = my_utils.create_video_to_id_mapping(
            [
                video_file.name
                for video_file in Path(DetectionPaths.videos_input).iterdir()
                if video_file.suffix.lower() == ".mp4"
            ]
        )
        # Create a shared annotation_id variable
        self.annotation_id = Value("i", 0)

    def call_models(
        self,
        detection_type,
        detection_function,
        video_file,
        file_name,
        models,
        output,
    ) -> dict:
        """
        This function performs a specific detection type on a video file.

        Parameters
        ----------
        detection_type : str
            the type of detection to perform
        detection_function : function
            the function to perform the detection
        video_file : str
            the video file to process
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
            category_id = LabelDictionaries.label_dict[detection_type]
            supercategory = LabelDictionaries.supercategory_dict[category_id]
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
                "person": {
                    "video_file_name": file_name,
                    "file_id": file_id,
                    "model": models.get(detection_type),
                },
                "face": {
                    "video_file_name": file_name,
                    "file_id": file_id,
                    "model": models.get(detection_type),
                },
                "voice": {},
            }
            # Call the function with the appropriate arguments
            detections = detection_function(
                video_file, self.annotation_id, **args.get(detection_type, {})
            )
            # Add detection_output to "images" and "annotations"
            output["images"].extend(detections["images"])
            output["annotations"].extend(detections["annotations"])

        elif detection_type == "proximity":
            logging.info("Performing proximity detection...")
            detection_function()

        return output

    def process_video_file(self, video_file, detections, models):
        """
        This function processes a video file by performing the specified detections.

        Parameters
        ----------
        video_file : str
            the video file to process
        detections : dict
            the detections to perform
        models : dict
            the models to use for detection
        """
        file_name = video_file.name
        file_name_short = video_file.stem
        video_file = str(video_file)
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
                "file_name": f"{file_name_short}.mp4",
            }
        )
        # Perform the desired detections
        for detection_type, detection_function in self.detection_functions.items():
            if detections[detection_type]:
                output = self.call_models(
                    detection_type,
                    detection_function,
                    video_file,
                    file_name,
                    models,
                    output,
                )
        # Save the result to a JSON file
        json_output_path = os.path.join(
            DetectionPaths.results, f"{file_name_short}_detections.json"
        )
        my_utils.save_results_to_json(output, json_output_path)

    def wrapper(self, args) -> None:
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
            if video_f.suffix.lower() == ".mp4"
        ]

        # Process each video file (in parallel)
        with ThreadPool() as pool:
            pool.map(
                self.wrapper,
                [(video_file, detections, models_dict) for video_file in video_files],
            )


def main(detections_dict: dict) -> None:
    """
    The main function of the social interactions detection pipeline

    Parameters
    ----------
    detections_dict : dict
        a dictionary indicating which detection models to run
        (person, face, voice, proximity)

    """
    start_time = timer()
    detector = Detector()
    detector.run_detection(detections_dict)
    end_time = timer()
    runtime = end_time - start_time
    logging.info(f"Runtime: {runtime} seconds")


if __name__ == "__main__":
    detections_dict = {
        "person": True,
        "face": True,
        "voice": True,
        "proximity": False,
    }
    main(detections_dict)

from multiprocessing import Pool
from pathlib import Path
from timeit import default_timer as timer
from persons import det_persons
from faces import my_mtcnn
from faces import my_fast_mtcnn
from language import detect_voices
from constants import DetectionParameters, DetectionPaths
import cv2
import my_utils
import logging
import functools
import os


# Set up logging
logging.basicConfig(level=logging.INFO)


class Detector:
    def __init__(self):
        self.detection_functions = {
            "person": det_persons.run_person_detection,
            "face": my_mtcnn.run_face_detection,
            "batch-wise face": my_fast_mtcnn.run_face_detection,
            "voice": detect_voices.extract_speech_duration,
            "proximity": my_utils.run_proximity_detection,
        }

        self.models = {
            "person": my_utils.load_person_detection_model,
            "face": my_utils.load_frame_face_detection_model,
            "batch-wise face": my_utils.load_batch_face_detection_model,
        }

    def get_detection_length(self, results, detection_type, video_file) -> int:
        """This function returns the length of the detection list for a given detection type.

        Parameters
        ----------
        results : _type_
            _description_
        detection_type : _type_
            _description_
        video_file : _type_
            _description_

        Returns
        -------
        int
            the length of the detection list
        """
        if detection_type in results:
            return len(results[detection_type])
        else:
            cap = cv2.VideoCapture(video_file)
            return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def call_models(
        self,
        detection_type,
        detection_function,
        video_file,
        file_name,
        models,
        results,
        coco_output,
    ) -> None:
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
        coco_output : dict
            the COCO output dictionary
        """
        logging.info(f"Performing {detection_type} detection...")
        detection_function = self.detection_functions[detection_type]
        if detection_type in ["person", "face", "batch-wise face"]:
            results[detection_type] = detection_function(
                video_file,
                file_name if detection_type in ["person", "face"] else None,
                models[detection_type],
                DetectionParameters.frame_step
                if detection_type in ["person", "face"]
                else None,
            )
        elif detection_type == "voice":
            len_detection_list = self.get_detection_length(
                results, detection_type, video_file
            )
            total_video_duration, voice_duration_sum, results["voice"] = (
                detection_function(video_file, len_detection_list)
            )
        elif detection_type == "proximity":
            logging.info("Performing proximity detection...")
            detection_function()

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
        # Initialize the results dictionary
        results = {}
        coco_output = {
            "videos": [],
            "images": [],
            "annotations": [],
            "categories": [],
        }
        for detection_type, detection_function in self.detection_functions.keys():
            if detections[detection_type]:
                self.call_models(
                    detection_type,
                    detection_function,
                    video_file,
                    file_name,
                    models,
                    coco_output,
                )
        # Save the result to a JSON file
        json_output_path = os.path.join(
            DetectionPaths.results, f"{file_name_short}.json"
        )
        my_utils.save_results_to_json(results, json_output_path)

    def run_detection(
        self,
        detections: dict,
    ) -> dict:
        """
        This function runs five different detection models:
        - Person detection
        - Face detection
        - Batch-wise face detection
        - Voice detection
        - Proximity detection
        For each detection model, the output is a list of 1s and 0s,
        where 1 indicates the presence of the object or event in the frame.

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
        models = {
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
        with Pool() as pool:
            pool.map(
                functools.partial(
                    self.process_video_file, detections=detections, models=models
                ),
                video_files,
            )


def main(detections_dict: dict) -> None:
    """
    The main function of the social interactions detection pipeline.

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
        "batch-wise face": False,  # "batch-wise face" detection is faster than "face" detection
        "voice": True,
        "proximity": False,
    }
    main(detections_dict)

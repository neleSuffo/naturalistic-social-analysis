import os
import json
import shutil
import torch
import logging
import config
from typing import Tuple
from facenet_pytorch import MTCNN
from fast_mtcnn import FastMTCNN
from language import detect_voices
from persons import det_persons
from faces import my_mtcnn
from faces import my_fast_mtcnn


def count_sequences(final_results, interaction_length):
    """
    This function counts the sequences of 2 or 3 in the final results.

    Parameters
    ----------
    final_results : list
        the final results list
    interaction_length : int
        the minimum length of the interaction sequence

    Returns
    -------
    list
        a list of sequence lengths of 2 or 3 in final_results
        that are greater than or equal to interaction_length
    """
    sequence_lengths = []
    sequence_length = 0

    for value in final_results:
        if value == 2 or value == 3:
            sequence_length += 1
        elif value == 0 or value == 1:
            if sequence_length >= interaction_length:
                sequence_lengths.append(sequence_length)
            sequence_length = 0

    return sequence_lengths


def run_person_detection(
    video_input_path: str, video_file_name: str, person_detection_model: torch.nn.Module
) -> list:
    """
    This function loads a video and performs person detection on it.

    Parameters
    ----------
    video_input_path : str
        the path to the video file
    video_file_name : str
        the name of the video file
    person_detection_model : torch.nn.Module
        the person detection model

    Returns
    -------
    list
        the results for each frame
        (1 if a person is detected, 0 otherwise)
    """
    # Set the output path for the person detection results
    output_path = os.path.join(config.video_person_output_path, video_file_name)
    return det_persons.run_person_detection(
        video_input_path, output_path, person_detection_model
    )


def run_frame_face_detection(
    video_input_path: str,
    video_file_name: str,
    face_detection_model: MTCNN,
) -> list:
    """
    This function loads a video and performs face detection on it.

    Parameters
    ----------
    video_input_path : str
        the path to the video file
    video_file_name : str
        the name of the video file
    face_detection_model : MTCNN
        the MTCNN face detector

    Returns
    -------
    list
        the results for each frame
        (1 if a face is detected, 0 otherwise)
    """
    # Set the output path for the face detection results
    output_path = os.path.join(config.video_face_output_path, video_file_name)
    return my_mtcnn.run_face_detection(
        video_input_path, output_path, face_detection_model
    )


def run_batch_face_detection(
    video_input_path: str,
    video_file_name: str,
    face_detection_model: MTCNN,
) -> list:
    """
    This function loads a video and performs face detection on it.

    Parameters
    ----------
    video_input_path : str
        the path to the video file
    video_file_name : str
        the name of the video file
    face_detection_model : MTCNN
        the MTCNN face detector

    Returns
    -------
    list
        the results for each frame
        (1 if a face is detected, 0 otherwise)
    """
    # Set the output path for the face detection results
    return my_fast_mtcnn.run_face_detection(video_input_path, face_detection_model)


def run_voice_detection(
    video_input_path: str, number_of_frames: int
) -> Tuple[float, float, list]:
    """
    This function loads a video and performs voice detection on it.

    Parameters
    ----------
    video_input_path : str
        the path to the video file
    number_of_frames : int
        the number of frames in the video

    Returns
    -------
    Tuple[float, float, list]
        the total duration of the video file,
        the total duration of the utterances in the video,
        the voice detection list indicating the presence of voice in each frame
        (1 if voice is present, 0 otherwise)
    """
    return detect_voices.extract_speech_duration(video_input_path, number_of_frames)


# TODO: Add proximity detection
def run_proximity_detection():
    pass  # Implement proximity detection here


def load_person_detection_model():
    """
    This function loads the person detection model.

    Returns
    -------
    torch.nn.Module
        the person detection model
    """
    # Load the YOLOv5 model for person detection
    person_detection_model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    # Move the model to the desired location
    if os.path.exists("yolov5s.pt"):
        shutil.move(
            "yolov5s.pt",
            "/Users/nelesuffo/projects/leuphana-IPE/pretrained_models/yolov5s.pt",
        )

    return person_detection_model


def load_batch_face_detection_model():
    """
    This function loads the face detection model.

    Returns
    -------
    MTCNN
        the face detection model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the batch MTCNN model
    face_detection_model = FastMTCNN(
        stride=2, resize=1, margin=14, factor=0.6, keep_all=True, device=device
    )
    return face_detection_model


def load_frame_face_detection_model():
    """
    This function loads the face detection model.

    Returns
    -------
    MTCNN
        the face detection model
    """
    # Load the full MTCNN model
    return MTCNN()


def save_results_to_json(results: dict, output_path: str) -> None:
    """
    This function saves the results to a JSON file.

    Parameters
    ----------
    results : dict
        the results for each detection type
    output_path : str
        the path to the output file
    """
    directory = os.path.dirname(output_path)
    # Check if the output directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Save the results to a JSON file
    with open(output_path, "w") as f:
        json.dump(results, f)


def display_results(results: dict) -> None:
    """
    This function calculates the percentage of frames where the object is detected
    and prints the results.

    Parameters
    ----------
    results : dict
        the results for each detection type
    """
    for detection_type, detection_list in results.items():
        percentage = sum(detection_list) / len(detection_list) * 100
        logging.info(
            f"Percentages of at least one {detection_type} detected relative to the total frames: {percentage:.2f}"
        )
        logging.info(
            f"Total number of frames ({detection_type}): {len(detection_list)}"
        )

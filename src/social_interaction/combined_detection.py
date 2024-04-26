import json
import os
import time
import cv2
import my_utils
import torch
from facenet_pytorch import MTCNN
from faces import my_MTCNN
from language import detect_voices
from persons import det_persons
import config

# Start the timer
start_time = time.time()


def perform_social_interactions_detection(
    person_detection: bool = True,
    face_detection: bool = True,
    voice_detection: bool = True,
    proximity_detection: bool = True,
) -> None:
    """
    This function performs social interactions detection.
    First, it loads the person detection and face detection models.
    Then, it gets a list of all video files in the input folder.
    Finally, it processes each video file by running the detection models

    Parameters
    ----------
    person_detection : bool
        whether to perform person detection (default is True)
    face_detection : bool
        whether to perform face detection (default is True)
    voice_detection : bool
        whether to perform voice detection (default is True)
    proximity_detection : bool
        whether to perform proximity detection (default is True)
    """
    # Load the person detection and face detection models
    person_detection_model, face_detection_model = my_utils.load_detection_models(
        person_detection, face_detection
    )

    # Get a list of all video files in the folder
    video_files = [
        f for f in os.listdir(config.videos_input_path) if f.lower().endswith(".mp4")
    ]

    # Process each video file
    for video_file in video_files:
        video_path = os.path.join(config.videos_input_path, video_file)  # noqa: E501
        run_detection_models(
            video_path,
            video_file,
            person_detection_model,
            face_detection_model,
            person_detection,
            face_detection,
            voice_detection,
            proximity_detection,
        )


def run_detection_models(
    video_input_path: str,
    video_file_name: str,
    person_detection_model: torch.nn.Module,
    face_detection_model: MTCNN,
    person_detection: bool,
    face_detection: bool,
    voice_detection: bool,
    proximity_detection: bool,
) -> None:  # noqa: E125
    """
    This function runs the social interactions detection pipeline.
    It performs person, face, voice, and proximity detection on the video.
    The output of each detection is a list of 1s and 0s, where 1 indicates
    that the object is detected in the frame and 0 indicates that it is not.
    The final output is the percentage of frames
    where all objects are detected.

    Parameters
    ----------
    video_input_path : str
        the path to the video file
    video_file_name : str
        the name of the video file
    person_detection_model : torch.nn.Module
        the person detection model
    face_detection_model : MTCNN
        the face detection model
    person_detection : bool
        whether to perform person detection
    face_detection : bool
        whether to perform face detection
    voice_detection : bool
        whether to perform voice detection
    proximity_detection : bool
        whether to perform proximity detection
    """

    # Get the number of frames in the video
    cap = cv2.VideoCapture(video_input_path)
    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results = {}
    print("Starting social interactions detection pipeline...")
    if person_detection:
        print("Performing person detection...")
        # Set the output path for the person detection results
        output_path = os.path.join(config.video_person_output_path, video_file_name)
        # Perform person detection on the video
        results["person"] = det_persons.run_person_detection(
            video_input_path, output_path, person_detection_model
        )
        # Get the length of the person detection results
        detection_length = len(results["person"])
    if face_detection:
        print("Performing face detection...")
        # Set the output path for the face detection results
        output_path = os.path.join(config.video_face_output_path, video_file_name)
        # Perform face detection on the video
        results["face"] = my_MTCNN.run_face_detection(
            video_input_path, output_path, face_detection_model
        )
        detection_length = len(results["face"])
    if voice_detection:
        print("Performing voice detection...")
        # Helper variable: true if either person
        # or face detection is enabled
        # to ensure that the detection list has the same length
        # as the person and face detection
        if face_detection or person_detection:
            number_of_frames = detection_length
        # Perform voice detection on the video
        (
            total_video_duration,
            voice_duration_sum,
            results["voice"],
        ) = detect_voices.extract_speech_duration(video_input_path, number_of_frames)  # noqa: E501
    if proximity_detection:
        print("Performing proximity detection...")
        pass  # Perform proximity detection on the video

    # Save the results to a JSON file
    with open("output/results.json", "w") as f:
        json.dump(results, f)

    for detection_type, detection_list in results.items():
        my_utils.calculate_percentage_and_print_results(detection_list, detection_type)  # noqa: E501


if __name__ == "__main__":
    perform_social_interactions_detection(
        person_detection=True,
        face_detection=True,
        voice_detection=True,
        proximity_detection=False,
    )

# Stop the timer and print the runtime
end_time = time.time()
runtime = end_time - start_time
print(f"Runtime: {runtime} seconds")

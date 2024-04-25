import json
import os
import time

import cv2
import my_utils
from faces import my_MTCNN
from language import detect_voices
from persons import det_persons

from social_interaction import config

# Start the timer
start_time = time.time()


def run_social_interactions_detection(
    video_input_path: str,
    person_detection: bool = True,
    face_detection: bool = True,
    voice_detection: bool = True,
    proximity_detection: bool = True,
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
    """

    # Get the number of frames in the video
    cap = cv2.VideoCapture(video_input_path)
    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results = {}
    print("Starting social interactions detection pipeline...")
    if person_detection:
        print("Performing person detection...")
        # Perform person detection on the video
        results["person"] = det_persons.person_detection(
            video_input_path, config.video_person_output_path
        )
        detection_length = len(results["person"])
    if face_detection:
        print("Performing face detection...")
        # Perform face detection on the video
        results["face"] = my_MTCNN.face_detection(
            video_input_path, config.video_face_output_path
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
        ) = detect_voices.extract_speech_duration(
            video_input_path, number_of_frames
        )  # noqa: E501
    if proximity_detection:
        print("Performing proximity detection...")
        pass  # Perform proximity detection on the video

    # Save the results to a JSON file
    with open("output/results.json", "w") as f:
        json.dump(results, f)

    for detection_type, detection_list in results.items():
        my_utils.calculate_percentage_and_print_results(
            detection_list, detection_type
        )  # noqa: E501


if __name__ == "__main__":
    # Get a list of all video files in the folder
    video_files = [
        f
        for f in os.listdir(config.videos_input_path)
        if f.lower().endswith(".mp4")  # noqa: E501
    ]
    # Process each video file
    for video_file in video_files:
        video_path = os.path.join(config.videos_input_path, video_file)  # noqa: E501
        run_social_interactions_detection(
            video_path,
            person_detection=True,
            face_detection=True,
            voice_detection=True,
            proximity_detection=False,
        )

# Stop the timer and print the runtime
end_time = time.time()
runtime = end_time - start_time
print(f"Runtime: {runtime} seconds")

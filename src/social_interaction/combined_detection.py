import os

import my_utils
from faces import my_MTCNN
from language import detect_voices
from persons import det_persons

from config import video_face_output_path  # noqa: E501
from config import video_person_output_path, videos_input_path


def run_social_interactions_detection(
    video_input_path: str,
    person_detection: bool = True,
    face_detection: bool = True,
    voice_detection: bool = True,
    proximity_detection: bool = True,
) -> None:  # noqa: E501
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
    results = {}
    if person_detection:
        # Perform person detection on the video
        results["person"] = det_persons.person_detection(
            video_input_path, video_person_output_path
        )  # noqa: E501
    if face_detection:
        # Perform face detection on the video
        results["face"] = my_MTCNN.face_detection(
            video_input_path, video_face_output_path
        )  # noqa: E501
    if voice_detection:
        # Perform voice detection on the video
        (
            total_video_duration,
            voice_duration_sum,
            results["voice"],
        ) = detect_voices.extract_speech_duration(video_input_path)
    if proximity_detection:
        pass  # Perform proximity detection on the video

    for detection_type, detection_list in results.items():
        my_utils.calculate_percentage_and_print_results(
            detection_list, detection_type
        )  # noqa: E501

        # Get the percentage of spoken language
        # relative to the length of the audio file
        # TODO: Adjust percentage calculation when more than one file is processed  # noqa: E501


if __name__ == "__main__":
    # Get a list of all video files in the folder
    video_files = [
        f
        for f in os.listdir(videos_input_path)
        if f.lower().endswith(".mp4")  # noqa: E501
    ]
    # Process each video file
    for video_file in video_files:
        video_path = os.path.join(videos_input_path, video_file)  # noqa: E501
        run_social_interactions_detection(
            video_path,
            person_detection=False,
            face_detection=True,
            voice_detection=True,
            proximity_detection=False,
        )

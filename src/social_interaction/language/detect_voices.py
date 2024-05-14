import os
from typing import Tuple
import cv2
from language import call_vtc
from language import config
from moviepy.editor import VideoFileClip
from language import my_utils


def extract_speech_duration(
    video_input_path: str,
    len_detection_list: int,
) -> Tuple[float, float, list]:
    """
    This function extracts the speech duration from a video file
    using the voice-type-classifier.
    First, it extracts the audio from the video and saves it as a WAV file.
    Then, it runs the voice-type-classifier on the audio file
    and converts the output to a pandas DataFrame.
    The dataframe is then saved as a parquet file.
    Lastly, it calculates the total duration of the utterances in the video.

    Parameters
    ----------
    video_input_path : str
        the path to the video file
    len_detection_list : int
        the length of the detection list from previous detections

    Returns
    -------
    float
        the total duration of the video file
    float
        the total duration of the utterances in the video
    list
        the voice detection list indicating the presence of voice in each frame
        (1 if voice is present, 0 otherwise)
    """
    # Load the video file and get the filename
    video = VideoFileClip(video_input_path)
    cap = cv2.VideoCapture(video_input_path)
    try:
        # Get the video properties
        total_video_duration = video.duration
        # Extract audio from the video and save it as a 16kHz WAV file
        file_name = os.path.basename(video_input_path)
        file_name_short = os.path.splitext(file_name)[0]
        my_utils.extract_resampled_audio(video, file_name)  # noqa: E501
    finally:
        cap.release()

    # Run the voice-type-classifier
    call_vtc.call_voice_type_classifier()

    # Delete the no longer needed audio file(s)
    my_utils.delete_files_in_directory(config.vtc_audio_path)

    # Convert the output of the voice-type-classifier to a pandas DataFrame
    vtc_output_df = my_utils.rttm_to_dataframe(config.vtc_output_file_path)

    # Get the total duration of the utterances
    utterance_duration_sum = my_utils.get_total_seconds_of_voice(
        vtc_output_df, file_name_short
    )

    # Generate a second-wise list of utterances
    second_wise_utterance_list = my_utils.generate_second_wise_utterance_list(
        total_video_duration,
        len_detection_list,
        vtc_output_df,
    )

    return (
        total_video_duration,
        utterance_duration_sum,
        second_wise_utterance_list,
    )

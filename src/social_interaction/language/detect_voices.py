import os
import sys
from typing import Tuple

import cv2
from language import call_vtc
from moviepy.editor import VideoFileClip

# Get the directory of my_utils.py and config.py and add it to the Python path
my_utils_dir = os.path.dirname(
    os.path.realpath(
        "/Users/nelesuffo/projects/leuphana-IPE/src/social_interaction/my_utils.py"  # noqa: E501
    )
)
sys.path.append(my_utils_dir)
import my_utils  # noqa: E402

config_dir = os.path.dirname(
    os.path.realpath("/Users/nelesuffo/projects/leuphana-IPE/src/config.py")
)  # noqa: E501
sys.path.append(config_dir)
from config import vtc_audio_path  # noqa: E402
from config import vtc_output_file_path  # noqa: E402


def extract_speech_duration(
    video_input_path: str,
    number_of_frames: int,
) -> Tuple[float, float, list]:  # noqa: E501
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
    number_of_frames : int
        the number of frames in the video

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
    with VideoFileClip(video_input_path) as video:
        # Get the video properties
        filename = os.path.basename(video_input_path)
        cap = cv2.VideoCapture(video_input_path)
        frames_per_second = cap.get(cv2.CAP_PROP_FPS)
        total_video_duration = video.duration
        # Extract audio from the video and save it as a 16kHz WAV file
        my_utils.extract_resampled_audio(video, filename)

    # Run the voice-type-classifier
    call_vtc.call_voice_type_classifier()

    # Delete the no longer needed audio file(s)
    my_utils.delete_files_in_directory(vtc_audio_path)

    # Convert the output of the voice-type-classifier to a pandas DataFrame
    vtc_output_df = my_utils.rttm_to_dataframe(vtc_output_file_path)

    # Get the total duration of the utterances
    utterance_duration_sum = my_utils.get_total_seconds_of_voice(vtc_output_df)

    # Generate a frame-wise list of utterances
    frame_wise_utterance_list = my_utils.generate_frame_wise_utterance_list(
        total_video_duration,
        frames_per_second,
        number_of_frames,
        vtc_output_df,  # noqa: E501
    )
    return (
        total_video_duration,
        utterance_duration_sum,
        frame_wise_utterance_list,
    )  # noqa: E501

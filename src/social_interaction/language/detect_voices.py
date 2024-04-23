import os
import subprocess
import sys

import pandas as pd
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
from config import videos_input_path  # noqa: E402
from config import vtc_environment_path  # noqa: E402
from config import vtc_execution_file_path  # noqa: E402
from config import vtc_output_file_path  # noqa: E402

pd.set_option("display.max_rows", None)


def extract_speech_duration(video_input_path: str) -> float:
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

    Returns
    -------
    float
        the total duration of the utterances in the video
    """
    # Load the video file and get the filename
    video = VideoFileClip(video_input_path)
    filename = os.path.basename(video_input_path)

    # Extract audio from the video and save it as a 16kHz WAV file
    my_utils.extract_resampled_audio(video, filename)

    # Define the path to the python executable and the command to run the voice-type-classifier  # noqa: E501
    env = os.environ.copy()
    env["PATH"] = (
        os.path.dirname(vtc_environment_path) + os.pathsep + env["PATH"]
    )  # noqa: E501

    # Run the voice-type-classifier using the voice-type-classifier environment
    subprocess.run([vtc_environment_path, vtc_execution_file_path], env=env)

    # Convert the output of the voice-type-classifier to a pandas DataFrame
    vtc_output_df = my_utils.rttm_to_dataframe(vtc_output_file_path)

    # Get the total duration of the utterances
    utterance_duration_sum = my_utils.total_seconds(vtc_output_df)

    # Get the total duration of the video file
    total_video_duration = my_utils.get_video_duration(video_input_path)
    return total_video_duration, utterance_duration_sum


if __name__ == "__main__":
    # Get a list of all video files in the folder
    video_files = [
        f for f in os.listdir(videos_input_path) if f.lower().endswith(".mp4")
    ]  # noqa: E501

    # Process each video file
    for video_file in video_files:
        video_path = os.path.join(videos_input_path, video_file)  # noqa: E501
        total_video_duration, utterance_duration_sum = extract_speech_duration(
            video_path
        )  # noqa: E501
        percentage = utterance_duration_sum / total_video_duration * 100
        print(
            f"Percentages of spoken language relative to the length of the audio file: {percentage:.2f}%"  # noqa: E501, E231
        )

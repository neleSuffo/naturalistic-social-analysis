import os
import sys
import subprocess
import pandas as pd
from moviepy.editor import VideoFileClip
# Get the directory of my_utils.py and config.py and add it to the Python path
my_utils_dir = os.path.dirname(os.path.realpath('/Users/nelesuffo/projects/leuphana-IPE/src/social_interaction/my_utils.py'))
sys.path.append(my_utils_dir)
import my_utils
config_dir = os.path.dirname(os.path.realpath('/Users/nelesuffo/projects/leuphana-IPE/src/config.py'))
sys.path.append(config_dir)
from config import videos_input_path, vtc_audio_path, vtc_execution_file_path, vtc_environment_path, vtc_output_file_path


def extract_speech_duration(video_input_path: str) -> pd.DataFrame:
    """
    This function extracts the speech duration from a video file using the voice-type-classifier.
    First, it extracts the audio from the video and saves it as a WAV file.
    Then, it runs the voice-type-classifier on the audio file and converts the output to a pandas DataFrame.

    Parameters
    ----------
    video_input_path : str
        the path to the video file

    Returns
    -------
    pd.DataFrame
        the output of the voice-type-classifier as a pandas DataFrame

    """
    # Load the video file and get the filename
    video = VideoFileClip(video_input_path)
    filename = os.path.basename(video_input_path)

    # Extract audio from the video and save it as a 16kHz WAV file
    my_utils.extract_resampled_audio(video, filename)

    # Define the path to the python executable and the command to run the voice-type-classifier
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(vtc_environment_path) + os.pathsep + env["PATH"]

    # Run the voice-type-classifier using the voice-type-classifier environment
    subprocess.run([vtc_environment_path, vtc_execution_file_path], env=env)

    # Convert the output of the voice-type-classifier to a pandas DataFrame
    vtc_output_df = my_utils.rttm_to_dataframe(vtc_output_file_path)

    # Get the total seconds of spoken language
    total_seconds_language = my_utils.total_seconds(vtc_output_df)
    return total_seconds_language

if __name__ == "__main__":
    # Get a list of all video files in the folder
    video_files = [f for f in os.listdir(videos_input_path) if f.lower().endswith('.mp4')]

    # Process each video file
    for video_file in video_files:
        video_path = os.path.join(videos_input_path, video_file)
        total_seconds_language = extract_speech_duration(video_path)
        print(total_seconds_language)

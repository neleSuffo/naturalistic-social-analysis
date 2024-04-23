import os
import sys

config_dir = os.path.dirname(
    os.path.realpath("/Users/nelesuffo/projects/leuphana-IPE/src/config.py")
)
sys.path.append(config_dir)
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from typing import Tuple  # noqa: E402

import cv2  # noqa: E402
import pandas as pd  # noqa: E402
from moviepy.editor import VideoFileClip  # noqa: E402

from config import vtc_audio_path, vtc_df_output_path  # noqa: E402


def total_seconds(df: pd.DataFrame) -> float:
    """
    This function calculates the total number of seconds covered by the intervals in a DataFrame.  # noqa: E501

    Parameters
    ----------
    df : pd.DataFrame
        the DataFrame containing the voice-type-classifier output

    Returns
    -------
    float
        the total number of seconds covered by the intervals in the DataFrame
    """
    # Sort the DataFrame by 'start'
    df = df.sort_values(by=["Utterance_Start"])

    # Initialize the total and the end of the last interval
    total = 0
    prev_end = 0
    df["Case"] = 0
    df["Seconds_Added"] = 0

    # Iterate over the intervals
    for row in df.itertuples():
        # Case 1: The utterance starts after the end of the last utterance
        if row.Utterance_Start > prev_end:
            # Add the full duration of the utterance to the total
            total += row.Utterance_Duration
            df.at[row.Index, "Case"] = 1
            df.at[row.Index, "Seconds_Added"] = row.Utterance_Duration
            prev_end = row.Utterance_End

        # Case 2: The utterance starts before the end of the last utterance
        # and ends after the end of the last utterance
        elif row.Utterance_Start <= prev_end and row.Utterance_End > prev_end:
            # Add the difference between the end of the utterance
            # and the end of the last utterance to the total
            total += row.Utterance_End - prev_end
            df.at[row.Index, "Case"] = 2
            df.at[row.Index, "Seconds_Added"] = row.Utterance_End - prev_end
            prev_end = row.Utterance_End
        # Save the output as a parquet file
        df.to_parquet(f"{vtc_df_output_path}vtc_output.parquet")
    return total


def rttm_to_dataframe(rttm_file: str) -> pd.DataFrame:
    """
    This function reads the voice_type_classifier
    output rttm file and returns its content as a pandas DataFrame.

    Parameters
    ----------
    rttm_file : str
        the path to the RTTM file

    Returns
    -------
    pd.DataFrame
        the content of the RTTM file as a pandas DataFrame
    """
    df = pd.read_csv(
        rttm_file,
        sep=" ",
        names=[
            "Speaker",
            "audio_file_name",
            "audio_file_id",
            "Utterance_Start",
            "Utterance_Duration",
            "NA_1",
            "NA_2",
            "Voice_type",
            "NA_3",
            "NA_4",
        ],
    )

    # Drop unnecessary columns
    df = df.drop(
        columns=["Speaker", "audio_file_id", "NA_1", "NA_2", "NA_3", "NA_4"]
    )  # noqa: E501
    df["Utterance_End"] = df["Utterance_Start"] + df["Utterance_Duration"]
    return df


def extract_resampled_audio(video: VideoFileClip, filename: str) -> None:
    """
    This function extracts the audio
    from a video file and saves it as a 16kHz WAV file.

    Parameters
    ----------
    video : VideoFileClip
        the video file
    filename : str
        the filename of the video
    """
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=True)

    # Extract the audio and save it to the temporary file
    video.audio.write_audiofile(temp_file.name + ".wav", codec="pcm_s16le")

    # Convert the audio to 16kHz with sox and save it to the output file  # noqa: E501
    # The temporary file will be deleted when the NamedTemporaryFile object is garbage collected  # noqa: E501
    output_file = os.path.join(
        vtc_audio_path, f"{filename[:-4]}_16kHz.wav"
    )  # noqa: E501
    subprocess.run(
        ["sox", temp_file.name + ".wav", "-r", "16000", output_file],
        check=True,  # noqa: E501
    )  # noqa: E501


def get_video_duration(file_path: str) -> float:
    """
    This function returns the duration of a video file.

    Parameters
    ----------
    file_path : str
        the path to the video file

    Returns
    -------
    float
        the duration of the video in seconds
    """
    clip = VideoFileClip(file_path)
    duration = clip.duration
    return duration


def get_video_properties(
    video_path: str,
) -> Tuple[cv2.VideoCapture, int, int, int, int]:  # noqa: E501
    """
    This function loads a video from a given path and returns its properties.

    Parameters
    ----------
    video_path : str
        the path to the video file

    Returns
    -------
    Tuple[cv2.VideoCapture, int, int, int, int]
        the video capture object, the width of the frame,
        the height of the frame, the number of frames
        in the video, the frames per second
    """
    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Get video properties of the video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = cap.get(cv2.CAP_PROP_FPS)

    return cap, frame_width, frame_height, frames_per_second


def create_video_writer(
    output_path: str,
    frames_per_second: int,
    frame_width: int,
    frame_height: int,  # noqa: E501
) -> cv2.VideoWriter:
    """
    This function creates a VideoWriter object to write the output video.

    Parameters
    ----------
    output_path : str
        the path to the output video file
    frames_per_second : int
        the frames per second of the video
    frame_width : int
        the width of the frame
    frame_height : int
        the height of the frame

    Returns
    -------
    cv2.VideoWriter
        the video writer object
    """

    # Create a VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path, fourcc, frames_per_second, (frame_width, frame_height)
    )

    return out

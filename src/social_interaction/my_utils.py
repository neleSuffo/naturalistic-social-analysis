import os
import subprocess
import sys
import tempfile
from typing import Tuple

import cv2
import pandas as pd
from moviepy.editor import VideoFileClip

config_dir = os.path.dirname(
    os.path.realpath("/Users/nelesuffo/projects/leuphana-IPE/src/config.py")
)
sys.path.append(config_dir)

from config import vtc_audio_path, vtc_df_output_path  # noqa: E402


def calculate_percentage_and_print_results(
    detection_list: list, detection_type: str
) -> None:  # noqa: E501
    """
    This function calculates the percentage of frames
    where the object is detected and prints the results.

    Parameters
    ----------
    detection_list : list
        the list of detections
    detection_type : str
        the type of detection
    """
    percentage = sum(detection_list) / len(detection_list) * 100
    print(
        f"Percentages of at least one {detection_type} detected relative to the total frames: {percentage:.2f}"  # noqa: E231, E501
    )
    print(f"Total number of frames ({detection_type}): {len(detection_list)}")


def delete_files_in_directory(directory_path: str):
    """
    This function deletes all files in a directory.

    Parameters
    ----------
    directory_path : str
        the path to the directory
    """
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.exists(file_path):
            os.remove(file_path)


def generate_frame_wise_utterance_list(
    total_video_duration: int,
    frames_per_second: float,
    total_nr_frames: int,
    df: pd.DataFrame,
) -> list:
    """
    This function generates a list indicating the presence
    of voice in each second of the video.
    1 indicates that voice is present, 0 indicates that it is not.

    Parameters
    ----------
    total_video_duration : int
        the total duration of the video in seconds
    frames_per_second : float
        the frames per second of the video
    total_nr_frames : int
        the total number of frames in the video
    df : pd.DataFrame
        the DataFrame containing the voice-type-classifier output

    Returns
    -------
    list
        the voice detection list indicating the presence of voice in each frame
        (1 if voice is present, 0 otherwise)
    """
    # Convert total_video_duration and frames_per_second to integers
    total_video_duration = int(total_video_duration)

    # Get video properties of the video
    frames_per_second = round(frames_per_second)

    # Initialize the second wise list with zeros and the frame wise list
    second_wise_utterance_list = [0] * total_video_duration
    frame_wise_utterance_list = []

    # Iterate over the utterances
    for row in df.itertuples():
        # Get the start and end times of the utterance in seconds
        start_time = int(row.Utterance_Start)
        end_time = int(row.Utterance_End)

        # Set the corresponding indices in the list to 1
        for i in range(start_time, end_time):
            second_wise_utterance_list[i] = 1

    # Convert the second wise list to a frame wise list
    for second in second_wise_utterance_list:
        # Repeat the value frame rate times and append to the new list
        frame_wise_utterance_list.extend([second] * frames_per_second)

    # Adjust the length of the expanded list
    # to match the exact number of frames
    if len(frame_wise_utterance_list) < total_nr_frames:
        # If the list is too short, append zeros
        frame_wise_utterance_list.extend(
            [0] * (total_nr_frames - len(frame_wise_utterance_list))
        )
    elif len(frame_wise_utterance_list) > total_nr_frames:
        # If the list is too long, truncate it
        frame_wise_utterance_list = frame_wise_utterance_list[:total_nr_frames]

    return frame_wise_utterance_list


def get_total_seconds_of_voice(df: pd.DataFrame) -> float:
    """
    This function calculates the total number of seconds covered by the intervals in a DataFrame.  # noqa: E501

    Parameters
    ----------
    df : pd.DataFrame
        the DataFrame containing the voice-type-classifier output

    Returns
    -------
    float
        the total number of seconds covered by the utterances in the DataFrame
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

    # Convert the audio to 16kHz with sox and
    # save it to the output file
    # The temporary file will be deleted when
    # the NamedTemporaryFile object is garbage collected
    output_file = os.path.join(vtc_audio_path, f"{filename[:-4]}_16kHz.wav")
    subprocess.run(
        ["sox", temp_file.name + ".wav", "-r", "16000", output_file],
        check=True,
    )


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
    nr_frames: bool = False,
) -> Tuple[cv2.VideoCapture, int, int, int, int, int]:  # noqa: E501
    """
    This function loads a video from a given path and returns its properties.

    Parameters
    ----------
    video_path : str
        the path to the video file
    nr_frames : bool
        whether to return the number of frames in the video, default is False

    Returns
    -------
    Tuple[cv2.VideoCapture, int, int, int, int]
        the video capture object, the width of the frame,
        the height of the frame, the number of frames
        in the video, the frames per second,
        and the number of frames in the video (if nr_frames is True)
    """
    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Get video properties of the video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = cap.get(cv2.CAP_PROP_FPS)
    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if nr_frames:
        return (
            cap,
            frame_width,
            frame_height,
            frames_per_second,
            number_of_frames,
        )  # noqa: E501

    return cap, frame_width, frame_height, frames_per_second


def create_video_writer(
    output_path: str,
    frames_per_second: int,
    frame_width: int,
    frame_height: int,
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

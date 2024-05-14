import os
import subprocess
import tempfile

import pandas as pd
from language import config
from moviepy.editor import VideoFileClip


def generate_second_wise_utterance_list(
    total_video_duration: int,
    len_detection_list: int,
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
    len_detection_list : float
        the length of the detection list from previous detections
    df : pd.DataFrame
        the DataFrame containing the voice-type-classifier output

    Returns
    -------
    list
        the voice detection list indicating the presence of voice in every second
        (1 if voice is present, 0 otherwise)
    """
    # Convert total_video_duration to integers
    total_video_duration = int(total_video_duration)

    # Initialize the second wise list with zeros and the frame wise list
    second_wise_utterance_list = [0] * total_video_duration

    # Iterate over the utterances
    for row in df.itertuples():
        # Get the start and end times of the utterance in seconds
        start_time = int(row.Utterance_Start)
        end_time = int(row.Utterance_End)

        # Set the corresponding indices in the list to 1
        for i in range(start_time, end_time):
            second_wise_utterance_list[i] = 1

    # Ensure the length of the list matches the detection list from previous detections
    if len(second_wise_utterance_list) >= len_detection_list:
        second_wise_utterance_list = second_wise_utterance_list[:len_detection_list]
    else:
        second_wise_utterance_list.extend(
            [0] * (len_detection_list - len(second_wise_utterance_list))
        )

    return second_wise_utterance_list


def generate_frame_wise_utterance_list(
    total_video_duration: int,
    frames_per_second: float,
    total_nr_frames: int,
    df: pd.DataFrame,
) -> list:
    """
    This function generates a list indicating the presence
    of voice in each frame of the video.
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
    output_file = os.path.join(config.vtc_audio_path, f"{filename[:-4]}_16kHz.wav")  # noqa: E501
    subprocess.run(
        ["sox", temp_file.name + ".wav", "-r", "16000", output_file],
        check=True,
    )


def get_total_seconds_of_voice(df: pd.DataFrame, file_name_short: str) -> float:
    """
    This function calculates the total number of seconds covered by the intervals in a DataFrame.  # noqa: E501

    Parameters
    ----------
    df : pd.DataFrame
        the DataFrame containing the voice-type-classifier output
    file_name_short: str
        the filename of the video

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
            df["Seconds_Added"] = df["Seconds_Added"].astype(float)
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
        parquet_output_path = os.path.join(
            config.vtc_df_output_path, f"{file_name_short}_vtc_output.parquet"
        )
        df.to_parquet(parquet_output_path)

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
    df = df.drop(columns=["Speaker", "audio_file_id", "NA_1", "NA_2", "NA_3", "NA_4"])  # noqa: E501
    df["Utterance_End"] = df["Utterance_Start"] + df["Utterance_Duration"]
    return df


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

from src.social_interaction.constants import (
    VTCParameters,
    LabelDictionaries,
    DetectionParameters,
)
from moviepy.editor import VideoFileClip
import pandas as pd
import os
import subprocess
import multiprocessing
import tempfile
import shutil


def get_utterances_detection_output(
    df: pd.DataFrame,
    annotation_id: multiprocessing.Value,
) -> dict:
    """
    This function generates a dictionary where each key is a second in the audio file
    and the value is a list of utterances that occurred during that second.

    Parameters
    ----------
    df : pd.DataFrame
        the DataFrame containing the voice-type-classifier output
    annotation_id : multiprocessing.Value
        the annotation ID to assign to the detections

    Returns
    -------
    dict
        the detection output
    """
    # Initialize detection output
    detection_output = {
        "images": [],
        "annotations": [],
    }

    # Get category ID from label dictionary
    category_id = LabelDictionaries.label_dict[DetectionParameters.vtc_detection_class]

    # Iterate over the utterances
    for row in df.itertuples():
        # Get the start and end times of the utterance in seconds
        start_time = int(row.Utterance_Start)
        end_time = int(row.Utterance_End)

        # For each second from the start to the end of the utterance
        for second in range(start_time, end_time):
            frame_count = second * DetectionParameters.frame_step
            # Add image information to COCO output
            detection_output["annotations"].append(
                {
                    "id": annotation_id.value,
                    "image_id": frame_count,
                    "category_id": category_id,
                    "voice_type": row.Voice_type,
                    "start_time": row.Utterance_Start,
                    "end_time": row.Utterance_End,
                }
            )
            with annotation_id.get_lock():
                annotation_id.value += 1

    return detection_output


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

    # Create the output directory if it doesn't exist
    os.makedirs(VTCParameters.audio_path, exist_ok=True)

    # Convert the audio to 16kHz with sox and
    # save it to the output file
    # The temporary file will be deleted when
    # the NamedTemporaryFile object is garbage collected
    output_file = os.path.join(VTCParameters.audio_path, f"{filename[:-4]}_16kHz.wav")  # noqa: E501
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
            VTCParameters.df_output_path, f"{file_name_short}_vtc_output.parquet"
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


def delete_files_and_directory(directory_path: str):
    """
    This function deletes all files and subdirectories
    in a directory, and then the directory itself.

    Parameters
    ----------
    directory_path : str
        the path to the directory
    """
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)

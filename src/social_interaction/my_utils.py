import cv2
from typing import Tuple
from moviepy.editor import VideoFileClip
import pandas as pd
import os


def total_seconds(df) -> float:
    """
    This function calculates the total number of seconds covered by the intervals in a DataFrame.

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
    df = df.sort_values(by='Utterance_Start')

    # Initialize the total and the end of the last interval
    total = 0
    last_end = -1

    # Iterate over the intervals
    for _, row in df.iterrows():
        # If this interval overlaps with the last one, extend the last interval
        # Otherwise, add the last interval to the total and start a new interval
        if row['Utterance_Start'] < last_end:
            last_end = max(last_end, row['Utterance_End'])
        else:
            total += last_end - row['Utterance_Start']
            last_end = row['Utterance_End']

    # Add the last interval to the total
    total += last_end - row['Utterance_Start']

    return total


def rttm_to_dataframe(rttm_file: str) -> pd.DataFrame:
    """
    This function reads the voice_type_classifier output rttm file and returns its content as a pandas DataFrame.

    Parameters
    ----------
    rttm_file : str
        the path to the RTTM file

    Returns
    -------
    pd.DataFrame
        the content of the RTTM file as a pandas DataFrame
    """
    df = pd.read_csv(rttm_file, 
                     sep=' ', 
                     names=['Speaker', 'audio_file_name', 'audio_file_id', 'Utterance_Start', 'Utterance_Duration', 
                            'NA_1', 'NA_2', 'Voice_type', 'NA_3', 'NA_4'])

    # Drop unnecessary columns
    df = df.drop(columns=['Speaker', 'audio_file_id', 'NA_1', 'NA_2', 'NA_3', 'NA_4'])
    df['Utterance_End'] = df['Utterance_Start'] + df['Utterance_Duration']
    return df


def extract_audio(video_input_path: str, 
                  audio_output_path: str) -> None:
    """
    This function extracts the audio from a video file and exports it as a WAV file.

    Parameters
    ----------
    video_input_path : str
        the path to the video file
    audio_output_path : str
        the path to export the audio as a WAV file
    """
    # Load the video and extract the audio
    video = VideoFileClip(video_input_path)
    audio = video.audio
    # Get the filename of the video
    filename = os.path.basename(video_input_path)
    # Export the audio as a WAV file

    audio.write_audiofile(audio_output_path + filename, codec='pcm_s16le')

def get_duration(file_path) -> float:
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

video_input_path = "/Users/nelesuffo/projects/leuphana-IPE/data/sample_1.MP4"
duration = get_duration(video_input_path)
print(f"The duration of the video is {duration} seconds.")

def get_video_properties(video_path: str) -> Tuple[cv2.VideoCapture, int, int, int, int]:
    """
    This function loads a video from a given path and returns its properties.

    Parameters
    ----------
    video_path : str
        the path to the video file

    Returns
    -------
    Tuple[cv2.VideoCapture, int, int, int, int]
        the video capture object, the width of the frame, the height of the frame, the number of frames in the video, the frames per second
    """
    # Load the video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties of the video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = cap.get(cv2.CAP_PROP_FPS)

    return cap, frame_width, frame_height, frames_per_second


def create_video_writer(output_path: str, 
                        frames_per_second: int,
                        frame_width: int,
                        frame_height: int) -> cv2.VideoWriter:
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, frames_per_second, (frame_width, frame_height))

    return out

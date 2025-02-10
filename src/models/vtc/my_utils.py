import pandas as pd
import subprocess
import tempfile
import logging
import json
from src.constants import VTCPaths
from src.config import VTCConfig, LabelToCategoryMapping, DetectionParameters, VideoConfig
from moviepy.editor import VideoFileClip
from pathlib import Path
from typing import Dict
from pyannote.core import Annotation, Segment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def convert_utterances_to_coco(
    df: pd.DataFrame,
    video_file_name : str,
) -> dict:
    """
    This function generates a dictionary where each key is a second in the audio file
    and the value is a list of utterances that occurred during that second.

    Parameters
    ----------
    df : pd.DataFrame
        the DataFrame containing the voice-type-classifier output
    video_file_name : str
        the name of the video file

    Returns
    -------
    dict
        the detection output
    """
    logging.info("Starting to generate detection output")
    
    # Initialize detection output
    detection_output = {
        "images": [],
        "annotations": [],
    }

    # Get category ID from label dictionary
    category_id = LabelToCategoryMapping.label_dict[DetectionParameters.vtc_detection_target]
    
    # Iterate over the utterances
    for row in df.itertuples():
        # Get the start and end times of the utterance in seconds
        start_time = int(row.Utterance_Start)
        end_time = int(row.Utterance_End)

        # For each second from the start to the end of the utterance
        for second in range(start_time, end_time):
            frame_count = second * DetectionParameters.frame_step_interval
            # Add image information to COCO output
            detection_output["images"].append(
                {
                "frame_id": frame_count, # frame number
                "file_name": f"{video_file_name}_{frame_count}.jpg",
                }
            )
            
            detection_output["annotations"].append(
                {
                    "category_id": category_id, # face category_id
                    "voice_type": row.Voice_type, # what type of voice is detected
                    "start_time": row.Utterance_Start, # start time of the utterance
                    "end_time": row.Utterance_End, # end time of the utterance
                }
            )
                
    logging.info("Detection output generation completed")
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
    VTCPaths.audio_dir.mkdir(parents=True, exist_ok=True)

    # Convert the audio to 16kHz with sox and
    # save it to the output file
    output_file = Path(VTCPaths.audio_dir) / f"{filename}{VTCConfig.audio_file_suffix}"
    subprocess.run(
        ["sox", temp_file.name + ".wav", "-r", "16000", output_file],
        check=True,
    )
    
    logging.info(f"Successfully stored the file at {output_file}")

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
        parquet_output_path = VTCPaths.df_output_pickle / f"{file_name_short}_vtc_output.parquet"
        df.to_parquet(parquet_output_path)

        return total


def rttm_to_dataframe(rttm_file: Path, output_path: Path) -> None:
    """
    This function reads the voice_type_classifier
    output rttm file and returns its content as a pandas DataFrame.

    Parameters
    ----------
    rttm_file : path
        the path to the RTTM file
    output_path : path
        the path to save the DataFrame
    """
    logging.info(f"Reading RTTM file from: {rttm_file}")
    
    try:
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
    except Exception as e:
        logging.error(f"Failed to read RTTM file: {e}")
        raise
    
    logging.info("Successfully read RTTM file. Processing data...")

    # Drop unnecessary columns
    df = df.drop(columns=["Speaker", "audio_file_id", "NA_1", "NA_2", "NA_3", "NA_4"])  # noqa: E501
    df["Utterance_End"] = df["Utterance_Start"] + df["Utterance_Duration"]
    
    logging.info("Data processing complete. Returning DataFrame.")

    try:
        df.to_pickle(output_path)
        logging.info(f"DataFrame successfully saved to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save DataFrame to file: {e}")
        raise
    
def convert_childlens_annotations_to_dataframe(data: Dict, fps: float) -> pd.DataFrame:
    """ 
    This function converts the annotations in a JSON file to a DataFrame.
    
    Parameters
    ----------
    data : Dict
        the data from the JSON file
    fps : float
        the frames per second of the video
    
    Returns
    -------
    pd.DataFrame
        the DataFrame containing the annotations
    """
    rows = []
    
    # Extract video ID, duration in seconds, and duration in frames
    video_id = data['metadata']['name']
    short_video_id = video_id.replace(".MP4", "")
    
    # Loop through each annotation instance
    for item in data['instances']:
        # Extract start and end time
        start_time = item["meta"]["start"]
        end_time = item["meta"]["end"]
        
        # Process each parameter and add its first annotation to the list
        for parameter in item.get("parameters", []):
            timestamps = parameter.get("timestamps", [])
            
            # Check if there is at least one timestamp
            if timestamps and "attributes" in timestamps[0] and timestamps[0]["attributes"]:
                # Collect all "name" entries in a list
                names = [attr["name"] for timestamp in timestamps for attr in timestamp.get("attributes", [])]

                # Initialize variables
                label = None
                gender = None
                age = None
                
                # Extract label, gender, and age
                for name in names:
                    if name in LabelToCategoryMapping.childlens_activities_to_include:
                        label = name
                    elif name in ["Male", "Female"]:
                        gender = name
                    elif name in ["Adult", "Child"]:
                        age = name
                        
                # Determine the new label based on the strategy
                new_label = determine_new_label(label, age, gender)
                
                # Add the annotation if a label was found
                if new_label is not None:
                    start_seconds = round(start_time / 1_000_000.0, 3)
                    end_seconds = round(end_time / 1_000_000.0, 3)
                    duration = round(end_seconds - start_seconds, 3)
                    
                    # Duplicate the row with "SPEECH" as the new label if applicable
                    if new_label in ["KCHI", "OCH", "MAL", "FEM"]:
                        rows.append({
                            "audio_file_name": short_video_id,
                            "Utterance_Start": start_seconds,
                            "Utterance_Duration": duration,
                            "Voice_type": "SPEECH",
                            "Utterance_End": end_seconds,
                        })
                    
                    # Append the row for this annotation
                    rows.append({
                        "audio_file_name": short_video_id,
                        "Utterance_Start": start_seconds,
                        "Utterance_Duration": duration,
                        "Voice_type": new_label,
                        "Utterance_End": end_seconds,
                    })
                    
    # Create a DataFrame from the rows
    df = pd.DataFrame(rows)
    return df

# Function to process all JSON files in a folder and convert to DataFrame
def process_superannotate_annotations_to_dataframe(folder_path: Path, fps: float = VideoConfig.fps) -> pd.DataFrame:
    """ 
    This function processes all JSON files in a folder and converts them to a single DataFrame.
    
    Parameters
    ----------
    folder_path : Path
        the path to the folder containing the JSON files
    fps : float
        the frames per second of the video
    
    Returns
    -------
    pd.DataFrame
        the combined DataFrame
    """
    
    all_dataframes = []
    
    # Iterate over all files in the specified folder
    for filename in folder_path.rglob("*.json"):
        logging.info(f"Processing file: {filename}")
        
        # Read the JSON file
        data = read_json(filename)
        
        # Convert annotations to DataFrame
        df = convert_childlens_annotations_to_dataframe(data, fps)
        all_dataframes.append(df)
    
    # Concatenate all DataFrames
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    logging.info("All files processed and combined into a single DataFrame")
    return combined_df


# Function to determine the new label based on the strategy
def determine_new_label(label: str, age: str, gender: str) -> str:
    """
    This function maps the original superannotate labels to the new labels.
    
    Parameters
    ----------
    label : str
        the original label
    age : str
        the age of the person
    gender : str
        the gender of the person
    
    Returns
    -------
    str
        the new label
    
    """
    # Map the original labels to the new labels
    if label in ["Child Talking", "Singing/Humming"]:
        return "KCHI"
    elif label == "Other Person Talking":
        if age == "Adult":
            if gender == "Female":
                return "FEM"
            elif gender == "Male":
                return "MAL"
        elif age == "Child":
            return "OCH"
    elif label == "Overheard Speech":
        return "SPEECH"
    return None

# Function to read JSON from a file
def read_json(file_path: str) -> Dict:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def dataframe_to_annotation(df: pd.DataFrame, label_column: str = "Voice_type") -> Annotation:
    """
    Converts a DataFrame to a pyannote.core.Annotation object.

    Parameters
    ----------
    df (pd.DataFrame)
        label_column (str): The column in the DataFrame that contains the labels.
    label_column (str): 
        The column in the DataFrame that contains the labels, default is "Voice_type".
    
    Returns
    -------
    Annotation
        the annotation object
    """
    annotation = Annotation()
    for _, row in df.iterrows():
        start = float(row["Utterance_Start"])
        end = float(row["Utterance_End"])
        label = row[label_column]
        annotation[Segment(start, end)] = label
    return annotation


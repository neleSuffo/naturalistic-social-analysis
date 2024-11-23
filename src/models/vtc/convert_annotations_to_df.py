import json
from pathlib import Path
from typing import Dict
import pandas as pd
import logging

# List of labels to include in the ActivityNet format
list_to_include = ['Child Talking',
                   'Other Person Talking',
                   'Overheard Speech',
                   'Singing/Humming',
                   ]

# Function to read JSON from a file
def read_json(file_path: str) -> Dict:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to determine the new label based on the strategy
def determine_new_label(label: str, age: str, gender: str) -> str:
    if label in ["Child Talking", "Singing/Humming"]:
        return "KCHI"
    elif label == "Other Person Talking":
        if age == "Adult":
            if gender == "Female":
                return "FEM"
            elif gender == "Male":
                return "MAL"
        elif age == "Child":
            return "CHI"
    elif label == "Overheard Speech":
        return "SPEECH"
    return None

# Conversion function
def convert_annotations_to_dataframe(data: Dict, fps: float = 30.0) -> pd.DataFrame:
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
                    if name in list_to_include:
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to process all JSON files in a folder and convert to DataFrame
def process_all_json_files_to_dataframe(folder_path: Path, fps: float = 30.0) -> pd.DataFrame:
    all_dataframes = []
    
    # Iterate over all files in the specified folder
    for filename in folder_path.glob("*.json"):
        logging.info(f"Processing file: {filename}")
        
        # Read the JSON file
        data = read_json(filename)
        
        # Convert annotations to DataFrame
        df = convert_annotations_to_dataframe(data, fps)
        all_dataframes.append(df)
    
    # Concatenate all DataFrames
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    logging.info("All files processed and combined into a single DataFrame")
    return combined_df

if __name__ == '__main__':
    input_dir = Path("/home/nele_pauline_suffo/ProcessedData/annotations_superannotate/annotations_Batch_1B/files")
    output_file = Path("/home/nele_pauline_suffo/ProcessedData/annotations_superannotate/quantex_share_annotations_1b")
    
    # Process all JSON files and convert to DataFrame
    combined_df = process_all_json_files_to_dataframe(input_dir)
    
    # Save the DataFrame to a pickle file
    combined_df.to_pickle(output_file.with_suffix('.pkl'))
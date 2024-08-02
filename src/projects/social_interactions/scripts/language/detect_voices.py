from src.projects.social_interactions.common.constants import VTCParameters
from src.projects.social_interactions.scripts.language import call_vtc, my_utils
import logging
import multiprocessing


def run_voice_detection(
    video_file_ids_dict: dict,
    annotation_id: multiprocessing.Value,
    image_id: multiprocessing.Value,
) -> dict:
    """
    This function runs the voice-type-classifier and generates the detection results in COCO format.

    Parameters
    ----------
    video_file_ids_dict : dict
        the dictionary containing the video file IDs
    annotation_id : multiprocessing.Value
        the unique annotation ID to assign to the detections
    image_id : multiprocessing.Value
        the unique image ID

    Returns
    -------
    dict
        the detection results in COCO format
    """
    # Initialize detection output
    detection_output = {
        "videos": [], 
        "annotations": [], 
        "images": []}

    # Create the output directory if it does not exist
    VTCParameters.output_path.mkdir(parents=True, exist_ok=True)
   
    try:
        # Run the voice-type-classifier
        call_vtc.run_voice_type_classifier_in_env()
    except Exception as e:
        logging.error(f"Failed to run voice-type-classifier: {e}")
    
    # Convert the output of the voice-type-classifier to a pandas DataFrame
    vtc_output_df = my_utils.rttm_to_dataframe(VTCParameters.output_file_path)

    # Get the unique audio file names
    unique_files = vtc_output_df['audio_file_name'].unique()
    VTCParameters.audio_name_ending

    for file_name in unique_files:
        # Get the video file name and file ID
        cleaned_audio_file_name = file_name.replace(VTCParameters.audio_name_ending.stem, '')
        file_id = video_file_ids_dict[cleaned_audio_file_name]
        
        # Filter DataFrame for current file
        file_df = vtc_output_df[vtc_output_df['audio_file_name'] == file_name]

        # Generate and append video entry
        detection_output["videos"].append({
            "id": file_id,
            "file_name": file_name,
        })
        # Generate detection output in COCO format
        file_detection_output = my_utils.get_utterances_detection_output(
            file_df,
            annotation_id,
            image_id,
            file_name,
            file_id,
        )
        # Accumulate annotations and images
        detection_output["annotations"].extend(file_detection_output["annotations"])
        detection_output["images"].extend(file_detection_output["images"])
    
    return detection_output

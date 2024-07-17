from src.projects.social_interactions.common.constants import VTCParameters
from src.projects.social_interactions.scripts.language import call_vtc, my_utils
from moviepy.editor import VideoFileClip
from pathlib import Path
import cv2
import logging
import multiprocessing


def run_voice_detection(
    video_input_path: Path,
    annotation_id: multiprocessing.Value,
    image_id: multiprocessing.Value,
    video_file_name: str,
    file_id: str,
) -> dict:
    """
    This function extracts the speech duration from a video file
    using the voice-type-classifier.
    First, it extracts the audio from the video and saves it as a WAV file.
    Then, it runs the voice-type-classifier on the audio file
    and outputs the detections in COCO format.

    Parameters
    ----------
    video_input_path : Path
        the path to the video file to process
    annotation_id : multiprocessing.Value
        the unique annotation ID to assign to the detections
    image_id : multiprocessing.Value
        the unique image ID
    video_file_name : str
        the name of the video file
    file_id: dict
        the unique video file id

    Returns
    -------
    dict
        the detection results in COCO format
    """
    # Create the output directory if it does not exist
    VTCParameters.output_path.mkdir(parents=True, exist_ok=True)
   
    # Load the video file and get the filename
    try:
        # Load the video file and get the filename
        video = VideoFileClip(str(video_input_path))
        cap = cv2.VideoCapture(str(video_input_path))
    except Exception as e:
        logging.error(f"Failed to load video file: {e}")
        raise

    try:
        # Extract audio from the video and save it as a 16kHz WAV file
        my_utils.extract_resampled_audio(video, video_input_path.name)
    except Exception as e:
        logging.error(f"Failed to extract audio from video: {e}")
        raise
    finally:
        cap.release()

    try:
        # Run the voice-type-classifier
        call_vtc.call_voice_type_classifier()
    except Exception as e:
        logging.error(f"Failed to run voice-type-classifier: {e}")
        raise

    # Convert the output of the voice-type-classifier to a pandas DataFrame
    vtc_output_df = my_utils.rttm_to_dataframe(VTCParameters.output_file_path)

    # Generate detection output in COCO format
    detection_output = my_utils.get_utterances_detection_output(
        vtc_output_df,
        annotation_id,
        image_id,
        video_file_name,
        file_id,
    )

    return detection_output

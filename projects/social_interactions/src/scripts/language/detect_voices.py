from projects.social_interactions.src.common.constants import VTCParameters
from moviepy.editor import VideoFileClip
from projects.social_interactions.src.scripts.language import call_vtc, my_utils
import cv2
import os
import logging


def run_voice_detection(
    video_input_path: str,
    annotation_id: int,
) -> dict:
    """
    This function extracts the speech duration from a video file
    using the voice-type-classifier.
    First, it extracts the audio from the video and saves it as a WAV file.
    Then, it runs the voice-type-classifier on the audio file
    and outputs the detections in COCO format.

    Parameters
    ----------
    video_input_path : str
        the path to the video file
    annotation_id : int
        the annotation ID to assign to the detections

    Returns
    -------
    dict
        the detection results in COCO format
    """
    # Load the video file and get the filename
    try:
        # Load the video file and get the filename
        video = VideoFileClip(video_input_path)
        cap = cv2.VideoCapture(video_input_path)

    except Exception as e:
        logging.error(f"Failed to load video file: {e}")
        raise
    try:
        # Extract audio from the video and save it as a 16kHz WAV file
        file_name = os.path.basename(video_input_path)
        my_utils.extract_resampled_audio(video, file_name)
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

    # Delete the no longer needed audio file(s) and the rttm files
    my_utils.delete_files_and_directory(VTCParameters.audio_path)
    my_utils.delete_files_and_directory(VTCParameters.output_path)

    # Generate detection output in COCO format
    detection_output = my_utils.get_utterances_detection_output(
        vtc_output_df,
        annotation_id,
    )

    return detection_output

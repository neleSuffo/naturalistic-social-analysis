from projects.social_interactions.src.common.constants import VTCParameters
from moviepy.editor import VideoFileClip
from pathlib import Path
from projects.social_interactions.src.scripts.language import call_vtc, my_utils
import cv2
import logging


def run_voice_detection(
    video_input_path: Path,
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
    video_input_path : Path
        the path to the video file to process
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

    # Delete all files and the directories
    my_utils.delete_directory_and_contents(VTCParameters.audio_path)
    my_utils.delete_directory_and_contents(VTCParameters.output_path)

    # Generate detection output in COCO format
    detection_output = my_utils.get_utterances_detection_output(
        vtc_output_df,
        annotation_id,
    )

    return detection_output

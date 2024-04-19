from persons import det_persons
from faces import dlib
import my_utils
import subprocess
import os


video_input_path = "/Users/nelesuffo/projects/leuphana-IPE/data/sample_1.MP4"
audio_output_path = '/Users/nelesuffo/projects/leuphana-IPE/voice_type_classifier/data/raw'
video_output_1_path = "runs/sample_1_det_person.mp4"


def run_social_interactions_detection(video_input_path: str,
                                      audio_output_path: str,
                                      video_output_1_path: str) -> None:
    """
    This function runs the combined detection of persons and spoken language in a video file.
    It then prints the percentages of person visible relative to the total frames and spoken language relative to the length of the audio file.

    Parameters
    ----------
    video_input_path : str
        the path to the video file
    audio_output_path : str
        the path to export the audio as a WAV file
    video_output_1_path : str
        the path to the output video file for person detection  
    """

    # Perform person detection on the video
    #person_detection_list = det_persons.person_detection(video_input_path, video_output_1_path)

    #dlib.detect_faces(video_input_path)
    #dlib.detect_faces(video_input_path)

    # Extract audio from the video and save it as a WAV file in the voice-type-classifier repository
    my_utils.extract_audio(video_input_path, audio_output_path)

    # Define the path to the python executable and the command to run the voice-type-classifier
    python_path = '/Users/nelesuffo/Library/Caches/pypoetry/virtualenvs/pyannote-afeazePz-py3.8/bin/python'
    command = '/Users/nelesuffo/projects/leuphana-IPE/voice_type_classifier/run_voice_type_classifier.py'
    env = os.environ.copy()
    env["PATH"] = os.path.dirname(python_path) + os.pathsep + env["PATH"]

    # Run the voice-type-classifier in the voice-type-classifier repository
    subprocess.run([python_path, command], env=env)
    
    # Convert the output of the voice-type-classifier to a pandas DataFrame
    voice_type_classifier_df = my_utils.rttm_to_dataframe('output_voice_type_classifier/data/all.rttm')
    
    # Calculate the total seconds of spoken language
    total_seconds_language = my_utils.total_seconds(voice_type_classifier_df)
    # Calculate the length of the audio file
    length_audio = my_utils.get_duration(video_input_path)
    # Print the percentages of person visible and spoken language
    #print(f'Percentages of person visbile relative to the total frames: {sum(person_detection_list) / len(person_detection_list)*100}')
    print(f'Percentages of spoken language relative to the length of the audio file: {total_seconds_language / length_audio*100}')
    print(total_seconds_language, length_audio)
if __name__ == "__main__":
    run_social_interactions_detection(video_input_path, audio_output_path, video_output_1_path)
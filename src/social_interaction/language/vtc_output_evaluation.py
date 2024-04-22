import os
import sys
import subprocess
# Get the directory of my_utils.py
my_utils_dir = os.path.dirname(os.path.realpath('path/to/my_utils.py'))
# Add the directory to the Python path
sys.path.append(my_utils_dir)
# Now you can import my_utils
import my_utils

OUTPUT_PATH = "/Users/nelesuffo/projects/leuphana-IPE/src/social_interaction/language/voice_type_classifier_df.csv"


def extract_speech_duration(video_input_path: str,
                                 audio_output_path: str):
    """
    

    """

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

    
    voice_type_classifier_df.to_csv(OUTPUT_PATH, index=False)
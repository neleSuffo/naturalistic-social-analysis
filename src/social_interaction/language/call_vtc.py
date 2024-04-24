import os
import subprocess
import sys

config_dir = os.path.dirname(
    os.path.realpath("/Users/nelesuffo/projects/leuphana-IPE/src/config.py")
)
sys.path.append(config_dir)
from config import vtc_environment_path  # noqa: E402
from config import vtc_execution_file_path  # noqa: E402


def call_voice_type_classifier():
    """
    This function calls the voice-type-classifier
    using the voice-type-classifier environment.
    """
    # Define the path to the python executable
    # and the command to run the voice-type-classifier
    env = os.environ.copy()
    env["PATH"] = (
        os.path.dirname(vtc_environment_path) + os.pathsep + env["PATH"]
    )  # noqa: E501

    # Run the voice-type-classifier
    # using the voice-type-classifier environment
    try:
        subprocess.run(
            [vtc_environment_path, vtc_execution_file_path], env=env
        )  # noqa: E501
    except Exception as e:
        print(
            f"An error occurred while running the voice-type-classifier: {e}"
        )  # noqa: E501
        raise

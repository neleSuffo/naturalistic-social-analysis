from projects.social_interactions.src.common.constants import VTCParameters
import os
import subprocess


def call_voice_type_classifier():
    """
    This function calls the voice-type-classifier
    using the voice-type-classifier environment.
    """
    # Define the path to the python executable
    # and the command to run the voice-type-classifier
    env = os.environ.copy()
    env["PATH"] = (
        os.path.dirname(VTCParameters.environment_path) + os.pathsep + env["PATH"]
    )
    env["PYTHONPATH"] = "/Users/nelesuffo/projects/leuphana-IPE"

    # Run the voice-type-classifier
    # using the voice-type-classifier environment
    try:
        subprocess.run(
            [VTCParameters.environment_path, VTCParameters.execution_file_path],
            env=env,  # noqa: E501
        )
    except Exception as e:
        print(f"An error occurred while running the voice-type-classifier: {e}")  # noqa: E501
        raise

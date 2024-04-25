import os
import subprocess

from language import config


def call_voice_type_classifier():
    """
    This function calls the voice-type-classifier
    using the voice-type-classifier environment.
    """
    # Define the path to the python executable
    # and the command to run the voice-type-classifier
    env = os.environ.copy()
    env["PATH"] = (
        os.path.dirname(config.vtc_environment_path) + os.pathsep + env["PATH"]
    )

    # Run the voice-type-classifier
    # using the voice-type-classifier environment
    try:
        subprocess.run(
            [config.vtc_environment_path, config.vtc_execution_file_path],
            env=env,  # noqa: E501
        )
    except Exception as e:
        print(
            f"An error occurred while running the voice-type-classifier: {e}"
        )  # noqa: E501
        raise

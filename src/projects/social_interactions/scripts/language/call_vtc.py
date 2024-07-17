from src.projects.social_interactions.common.constants import VTCParameters
import subprocess
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def call_voice_type_classifier():
    """
    This function calls the voice-type-classifier
    using the voice-type-classifier environment.
    """
    # Define the path to the python executable
    # and the command to run the voice-type-classifier
    env = os.environ.copy()
    env["PATH"] = (
        str(VTCParameters.environment_path.parent) + os.pathsep + env["PATH"]
    )
    env["PYTHONPATH"] = "/home/nele_pauline_suffo/projects/leuphana-IPE"
    
    # Run the voice-type-classifier
    # using the voice-type-classifier environment
    try:
        subprocess.run(
            [VTCParameters.environment_path, VTCParameters.execution_file_path],
            env=env,
            check=True  # This will raise an exception if the command fails
        )
    except Exception as e:
        logging.exception("An unexpected error occurred while running the voice-type-classifier")
        raise
    else:
        logging.info("Voice-type-classifier ran successfully.")

import os
import subprocess
import sys

config_dir = os.path.dirname(
    os.path.realpath("/Users/nelesuffo/projects/leuphana-IPE/src/config.py")
)  # noqa: E501
sys.path.append(config_dir)
from config import vtc_audio_path, vtc_execution_command  # noqa: E402


# Run the voice-type-classifier
def run_voice_type_classifier(input_dir: str) -> None:
    """
    This function runs the voice-type-classifier
    on the audio file in the input directory.

    Parameters
    ----------
    input_dir : str
       the path to the input directory with the audio file
    """
    # Run the voice-type-classifier on the input files
    subprocess.run(
        [vtc_execution_command, input_dir, "--device=gpu"], check=True
    )  # noqa: E501


if __name__ == "__main__":
    run_voice_type_classifier(vtc_audio_path)

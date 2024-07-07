from src.projects.social_interactions.common.constants import VTCParameters
import subprocess


def run_voice_type_classifier(input_dir: str) -> None:
    """
    This function runs the voice-type-classifier on the input files.

    Parameters
    ----------
    input_dir : str
        the path to the input directory with the audio file
    """
    # Run the voice-type-classifier on the input files and store the output
    subprocess.run(
        [VTCParameters.execution_command, input_dir, "--device=gpu"],
        check=True,
        stdout=subprocess.PIPE,
    )


if __name__ == "__main__":
    run_voice_type_classifier(VTCParameters.audio_path)

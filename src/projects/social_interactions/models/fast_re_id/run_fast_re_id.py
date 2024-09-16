import subprocess
from pathlib import Path
from src.projects.social_interactions.common.constants import FastReIDParameters as FRP
from src.projects.social_interactions.config.config import FastReIDConfig as FRC

def main():
    # Loop through each subdirectory in the base folder
    for folder in FRP.base_dir.iterdir():
        if folder.is_dir():
            input_files = str(folder / '*.jpg')
            output_subdir = FRP.output_dir / folder.name
            output_subdir.mkdir(parents=True, exist_ok=True)

            # Shell command to run the script with the proper environment
            command = f"""
            cd {FRP.python_env_path} && \
            poetry run python tools/deploy/trt_inference.py \
            --model-path {FRP.trt_engine_path} \
            --input {input_files} \
            --batch-size {FRC.trt_batch_size} \
            --height {FRC.trt_height} \
            --width {FRC.trt_width} \
            --output {output_subdir}
            """
            
            # Run the command
            subprocess.run(command, shell=True, check=True)

if __name__ == "__main__":
    main()
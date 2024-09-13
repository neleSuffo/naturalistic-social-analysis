import subprocess
from pathlib import Path
from src.projects.social_interactions.common.constants import FastReIDParameters as FRP


def main():
    # Loop through each subdirectory in the base folder
    for folder in FRP.base_folder.iterdir():
        if folder.is_dir():
            input_files = str(folder / '*.jpg')
            output_subdir = FRP.output_dir / folder.name
            output_subdir.mkdir(parents=True, exist_ok=True)

            # Shell command to run the script with the proper environment
            command = f"""
            cd {FRP.env} && \
            poetry run python tools/deploy/trt_inference.py \
            --model-path {FRP.trt_engine} \
            --input {input_files} \
            --batch-size {FRP.trt_batch_size} \
            --height {FRP.trt_height} \
            --width {FRP.trt_width} \
            --output {output_subdir}
            """
            
            # Run the command
            subprocess.run(command, shell=True, check=True)

if __name__ == "__main__":
    main()
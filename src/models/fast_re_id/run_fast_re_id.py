import subprocess
from pathlib import Path
from constants import FastReIDParameters
from config import FastReIDConfig

def main():
    # Loop through each subdirectory in the base folder
    for folder in FastReIDParameters.base_dir.iterdir():
        if folder.is_dir():
            input_files = str(folder / '*.jpg')
            output_subdir = FastReIDParameters.output_dir / folder.name
            output_subdir.mkdir(parents=True, exist_ok=True)

            # Shell command to run the script with the proper environment
            command = f"""
            cd {FastReIDParameters.python_env_path} && \
            poetry run python tools/deploy/trt_inference.py \
            --model-path {FastReIDParameters.trt_engine_path} \
            --input {input_files} \
            --batch-size {FastReIDConfig.trt_batch_size} \
            --height {FastReIDConfig.trt_height} \
            --width {FastReIDConfig.trt_width} \
            --output {output_subdir}
            """
            
            # Run the command
            subprocess.run(command, shell=True, check=True)

if __name__ == "__main__":
    main()
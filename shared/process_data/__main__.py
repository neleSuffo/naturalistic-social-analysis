# main.py

import subprocess
from shared.process_data.prepare_training import main as prepare_training

def run_process_annotations():
    # Run the process_annotations module
    try:
        subprocess.run(['python', '-m', 'shared.process_data.process_annotations.__main__'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running process_annotations: {e}")

def run_process_videos():
    # Run the process_videos module
    try:
        subprocess.run(['python', '-m', 'shared.process_data.process_videos.__main__'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running process_videos: {e}")

if __name__ == "__main__":
    run_process_annotations()
    run_process_videos()
    # Split the dataset into training and validation sets
    prepare_training()
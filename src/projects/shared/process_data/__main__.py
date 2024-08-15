import subprocess
import logging
from src.projects.shared.process_data.prepare_training import main as prepare_training

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_process_annotations():
    # Run the process_annotations module
    try:
        subprocess.run(['python', '-m', 'src.projects.shared.process_data.process_annotations.__main__'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running process_annotations: {e}")

def run_process_videos():
    # Run the process_videos module
    try:
        subprocess.run(['python', '-m', 'src.projects.shared.process_data.process_videos.__main__'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running process_videos: {e}")

if __name__ == "__main__":
    #run_process_videos()
    #run_process_annotations()
    # Split the dataset into training and validation sets
    prepare_training()
    logging.info("Data preparation complete.")

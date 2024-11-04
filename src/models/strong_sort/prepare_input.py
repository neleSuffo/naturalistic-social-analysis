import subprocess
import logging
from constants import StrongSortPaths, FastReIDPaths, BasePaths
from config import FastReIDConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_ecc_results():
    """
    This function generates the ECC results for the train and val videos.
    """
    # Command to activate environment and run the ECC script for train and val
    command_train = f"conda run -n strongsort python {StrongSortPaths.ecc_script} --mot_dir {StrongSortPaths.train_videos_dir} --output_path {StrongSortPaths.ecc_train_output_path}"
    command_val = f"conda run -n strongsort python {StrongSortPaths.ecc_script} --mot_dir {StrongSortPaths.val_videos_dir} --output_path {StrongSortPaths.ecc_val_output_path}"

    # Execute the train command
    logging.info(f"Generating ECC results for train videos {StrongSortPaths.train_videos_dir}.")
    process_train = subprocess.Popen(command_train, shell=True, executable='/bin/bash')
    process_train.communicate()  # Wait for train process to complete

    # Execute the val command
    logging.info(f"Generating ECC results for val videos {StrongSortPaths.val_videos_dir}.")
    process_val = subprocess.Popen(command_val, shell=True, executable='/bin/bash')
    process_val.communicate()  # Wait for val process to complete
    
    logging.info("ECC alignment complete for both train and val datasets.")


def generate_fast_re_id_features():  
    """
    This function generates the FastReID features for all videos.
    """
    def generate_re_id_features_per_video(video_name: str):
        """
        This function generates the FastReID features for a given video.
        """
        # Set the input path  
        input_path = f"{video_name}/*jpg"
        
        # Check if the folder exists in either train or val directories
        output_dir_train = StrongSortPaths.train_videos_dir/ video_name.name
        output_dir_val = StrongSortPaths.val_videos_dir/ video_name.name
        
        if output_dir_train.exists():
            output_path = output_dir_train
        elif output_dir_val.exists():
            output_path = output_dir_val
        else:
            raise FileNotFoundError(f"Neither {output_dir_train} nor {output_dir_val} exists.")
        python_executable = FastReIDPaths.python_env_path/"bin/python"
        model_path = "/home/nele_pauline_suffo/models/duke_R101.engine"
        
        command = [
            python_executable,
            "tools/deploy/trt_inference.py",
            "--model-path", model_path,
            "--input", input_path,
            "--batch-size", str(FastReIDConfig.trt_batch_size),
            "--height", str(FastReIDConfig.trt_height),
            "--width", str(FastReIDConfig.trt_width),
            "--output", output_path
]
        # Run the command
        #subprocess.run(command, shell=True, executable='/bin/bash', check=True)
        try:
            subprocess.run(command, cwd=str(BasePaths.fast_re_id_dir), check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            
    for video_subfolder in FastReIDPaths.images_train_dir.iterdir():
        logging.info(f"Found {len(list(FastReIDPaths.images_train_dir.iterdir()))} video(s) in train.")
        generate_re_id_features_per_video(video_subfolder)
        logging.info(f"Generated FastReID features for {video_subfolder}.")
        
    for video_subfolder in FastReIDPaths.images_val_dir.iterdir():
        logging.info(f"Found {len(list(FastReIDPaths.images_val_dir.iterdir()))} video(s) in val.")
        generate_re_id_features_per_video(video_subfolder)
        logging.info(f"Generated FastReID features for {video_subfolder}.")

if __name__ == "__main__":
    # 1. Generate ECC results for the train and val videos
    #generate_ecc_results()
    
    # 2. Generate FastReID features for all videos
    generate_fast_re_id_features()

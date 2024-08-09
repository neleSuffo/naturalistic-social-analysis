import shutil
import random
import logging
import shutil
import os
import sqlite3
import subprocess
from pathlib import Path
from src.projects.social_interactions.common.constants import DetectionPaths, TrainParameters, YoloParameters as Yolo, MtcnnParameters as Mtcnn, DetectionParameters

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_missing_annotation_files(
    jpg_dir: Path, 
    txt_dir: Path
) -> None:
    """
    This function checks if for every .jpg file there is a .txt file with the same name. 
    If not, it creates an empty .txt file.

    Parameters
    ----------
    jpg_dir : Path
        the directory with the .jpg files
    txt_dir : Path
        the directory with the .txt files
    """
    # Get the list of all .jpg and .txt files
    jpg_files = {f.stem for f in jpg_dir.glob('*.jpg')}
    txt_files = {f.stem for f in txt_dir.glob('*.txt')}
    
    # Find the missing .txt files
    missing_txt_files = jpg_files - txt_files
    
    # Create empty .txt files for the missing .jpg files
    for file in missing_txt_files:
        (txt_dir / f"{file}.txt").touch()


def split_dataset(
    images_dir: Path,
    train_videos: list,
    val_videos: list,
    ) -> tuple:
    """
    This function splits the dataset into training and validation sets
    and returns the list of training and validation files.

    Parameters
    ----------
    images_dir : str
        the directory with the .jpg files
    train_videos : list
        the list of training videos
    val_videos : list
        the list of validation videos
    
    Returns
    -------
    tuple
        the list of training and validation images
    """
    # Initialize empty lists to store image names for train and validation sets
    train_images = []
    val_images = []

    # Iterate through all images in the directory
    for image_file in images_dir.glob("*.jpg"):  # Assuming images are in JPG format
        # Get the image file name without extension
        image_base_name = image_file.stem
        
        # Find the video name part (before the last underscore and digits)
        video_name_part = "_".join(image_base_name.split("_")[:-1])
        
        # Check if the video name part matches any video in the train or validation sets
        if video_name_part in train_videos:
            train_images.append(image_file.name)
        elif video_name_part in val_videos:
            val_images.append(image_file.name)
    
    logging.info(f"Dataset split completed. Training files: {len(train_images)}, Validation files: {len(val_images)}")
    return train_images, val_images


def copy_yolo_files(
    files_to_move_lst: list, 
    dest_dir_images: Path,
    dest_dir_labels: Path,
    ) -> None:
    """
    This function moves files from the source directory to the destination directory
    and deletes the empty source directory.

    Parameters
    ----------
    files_to_move_lst : list
        the list of files to move
    dest_dir_images : Path
        the destination directory for the image files
    dest_dir_labels : Path
        the destination directory for the label files
    """        
    logging.info(f"Moving YOLO files to {dest_dir_images} and {dest_dir_labels}")
    # Move the files to the new directory
    for file_path in files_to_move_lst:
        # Construct the full source paths for the image and label
        src_label_path = Yolo.labels_input / file_path.with_suffix('.txt').name

        # Construct the destination paths for the image and label
        dest_image_path = dest_dir_images / file_path.name
        dest_label_path = dest_dir_labels / file_path.with_suffix('.txt').name

        # Copy the images and move the label to their new destinations
        shutil.copy(file_path, dest_image_path)
        src_label_path.rename(dest_label_path)
    logging.info("YOLO files copied successfully")


def prepare_yolo_dataset(
    destination_dir: Path,
    train_files: list,
    val_files: list,
):
    """
    This function moves the training and validation files to the new yolo directories.

    Parameters
    ----------
    destination_dir : Path
        the destination directory to store the training and validation sets
    train_files : list
        the list of training files
    val_files : list
        the list of validation files
    """
    # Define source directory and new directories for training
    train_dir_images = destination_dir / "images/train"
    val_dir_images = destination_dir / "images/val"
    train_dir_labels = destination_dir / "labels/train"
    val_dir_labels = destination_dir / "labels/val"
    
    # Create necessary directories if they don't exist
    for path in [train_dir_images, val_dir_images, train_dir_labels, val_dir_labels]:
        path.mkdir(parents=True, exist_ok=True)
    
    # Move the files to the new directories
    copy_yolo_files(train_files, train_dir_images, train_dir_labels)
    copy_yolo_files(val_files, val_dir_images, val_dir_labels)  
    
    # Delete the empty labels directory
    shutil.rmtree(Yolo.labels_input)


def copy_mtcnn_files(
    train_files: list,
    val_files: list, 
    train_dir_images: Path,
    val_dir_images: Path,
    train_labels_path: Path,
    val_labels_path: Path,
    )-> None:
    """
    This function moves the training and validation files to the new mtcnn directories.

    Parameters
    ----------
    train_files : list
        the list of training files
    val_files : list
        the list of validation files
    train_dir_images: Path
        the directory to store the training images
    val_dir_images: Path
        the directory to store the validation images
    train_labels_path: Path
        the path to store the training labels
    val_labels_path: Path
        the path to store the validation labels
    """
    logging.info(f"Moving MTCNN files to {train_dir_images}, {val_dir_images}, {train_labels_path} and {val_labels_path}")
    # Move the images to the training and validation directories
    for file_path in train_files:
        src_image_path = DetectionPaths.images_input / file_path.name
        dest_image_path = train_dir_images / file_path.name
        shutil.copy(src_image_path, dest_image_path)
    for file_path in val_files:
        src_image_path = DetectionPaths.images_input / file_path.name
        dest_image_path = val_dir_images / file_path.name
        shutil.copy(src_image_path, dest_image_path)
    
    # Convert lists to sets and extract the file names
    train_set = set(train_files)
    val_set = set(val_files)
    file_names_in_train_set = {path.name.rsplit('.', 1)[0] for path in train_set}
    file_names_in_val_set = {path.name.rsplit('.', 1)[0] for path in val_set}

    # Move the labels to the training and validation directories
    with Mtcnn.labels_input.open('r') as original_file, train_labels_path.open('w') as train_file, val_labels_path.open('w') as validation_file:
        for line in original_file:
            image_file_name = line.split()[0]  # Assuming the file name is the first element
            if image_file_name in file_names_in_train_set:
                train_file.write(line)
            elif image_file_name in file_names_in_val_set:
                validation_file.write(line)
    logging.info("MTCNN files copied successfully")


def prepare_mtcnn_dataset(
    destination_dir: Path,
    train_files: list,
    val_files: list,
):
    """
    This function moves the training and validation files to the new yolo directories.

    Parameters
    ----------
    destination_dir : Path
        the destination directory to store the training and validation sets
    train_files : list
        the list of training files
    val_files : list
        the list of validation files
    """
    # Define source directory and new directories for training
    train_dir_images = destination_dir / "train/images"
    train_labels_path = destination_dir / "train/train.txt"
    val_dir_images = destination_dir / "val/images"
    val_labels_path = destination_dir / "val/val.txt"

    # Create necessary directories if they don't exist
    for path in [train_dir_images, val_dir_images, train_labels_path.parent, val_labels_path.parent]:
        path.mkdir(parents=True, exist_ok=True)
    
    # Move the files to the new directories
    copy_mtcnn_files(train_files, 
                    val_files, 
                    train_dir_images, 
                    val_dir_images,
                    train_labels_path, 
                    val_labels_path,
    )
    
    # Delete the empty labels directory
    Mtcnn.labels_input.unlink(missing_ok=True)

def get_video_length(
    file_path: Path
) -> float:
    """
    This function returns the length of a video file in seconds.
    
    Parameters
    ----------
    file_path : Path
        the path to the video file
    
    Returns
    -------
    float
        the length of the video in seconds
    """
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    return float(result.stdout)


def get_video_lengths() -> list:
    """
    This function returns a list of video files with their lengths in seconds.

    Returns
    -------
    list
        the list of video files with their lengths
    """
    # Initialize the list to store the video names and lengths
    video_lengths = []
    
    # Connect to the annotations database
    conn = sqlite3.connect(DetectionPaths.annotations_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT file_name FROM videos")
    video_files_list = [row[0] for row in cursor.fetchall()]
    
    # Iterate over the video files
    for video in video_files_list:
        # Construct the full path to the video file
        video_name = os.path.splitext(video)[0]
        possible_files = list(DetectionPaths.videos_input.glob(f"{video_name}.mp4")) + list(DetectionPaths.videos_input.glob(f"{video_name}.MP4"))
        if not possible_files:
            print(f"File not found for {video_name}, skipping...")
            continue
        # Take the first match found
        video_path = possible_files[0]  
        # Calculate the length of the video
        length = get_video_length(video_path)
        # Append the video name and length to the list
        video_lengths.append((video_name, length))
    
    return video_lengths


def balanced_random_train_val_split(
    train_ratio: float
) -> tuple:
    
    """
    This function splits the dataset into training and validation sets
    while balancing the cumulative length of the videos in each set.
    
    Parameters
    ----------
    train_ratio : float
        the ratio to split the dataset
        
    Returns
    -------
    tuple
        the list of training and validation videos
    """
    # Get the list of video files and their lengths
    video_files_with_length = get_video_lengths()

    # Set the random seed for reproducibility
    random.seed(TrainParameters.random_seed)
    # Shuffle the list to randomize the selection
    random.shuffle(video_files_with_length)
    
    # Initialize training and validation sets
    train_videos = []
    val_videos = []

    # Initialize cumulative lengths
    train_length = 0
    val_length = 0
    
    # Determine the target number of training videos based on the train_ratio
    total_videos = len(video_files_with_length)
    num_train_videos = int(train_ratio * total_videos)
    num_val_videos = total_videos - num_train_videos

    # Distribute videos to balance cumulative length
    for video in video_files_with_length:
        if len(train_videos) < num_train_videos and (train_length <= val_length or len(val_videos) >= num_val_videos):
            train_videos.append(video[0])  # Append only the video name
            train_length += video[1]
        else:
            val_videos.append(video[0])  # Append only the video name
            val_length += video[1]
    
    # Calculate average lengths
    train_avg_length = train_length / len(train_videos) if train_videos else 0
    val_avg_length = val_length / len(val_videos) if val_videos else 0
    
    # Print the results
    print(f"Training set: {len(train_videos)} videos, Average Length: {train_avg_length:.2f} seconds")
    print(f"Validation set: {len(val_videos)} videos, Average Length: {val_avg_length:.2f} seconds")

    return train_videos, val_videos

def main():
    os.environ['OMP_NUM_THREADS'] = '1'
    create_missing_annotation_files(DetectionPaths.images_input, Yolo.labels_input)
    # Split video files into training and validation sets
    train_videos, val_videos  = balanced_random_train_val_split(TrainParameters.train_test_split) 
    # Split corresponding image files into training and validation sets
    train_files, val_files = split_dataset(DetectionPaths.images_input, train_videos, val_videos)
    # Move label files and delete empty labels directory
    prepare_yolo_dataset(Yolo.data_input, train_files, val_files)
    prepare_mtcnn_dataset(Mtcnn.data_input, train_files, val_files)


if __name__ == "__main__":
    main()

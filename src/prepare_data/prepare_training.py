import shutil
import random
import logging
import os
import sqlite3
import subprocess
from pathlib import Path
from constants import DetectionPaths, YoloPaths, MtcnnPaths
from config import TrainingConfig
import argparse
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
    # Get the list of all .jpg and .txt files and extract the file names
    jpg_files = {f.stem for f in jpg_dir.rglob('*.jpg')}
    txt_files = {f.stem for f in txt_dir.glob('*.txt')}
    missing_txt_files = jpg_files - txt_files
    
    # Create empty .txt files for the missing .jpg files
    for file in missing_txt_files:
        (txt_dir / f"{file}.txt").touch()


def images_train_val_test_split(
    images_dir: Path,
    train_videos: list,
    val_videos: list,
    test_videos: list
    ) -> tuple:
    """
    This function splits the dataset into training, validation and testing sets
    and returns the list of training, validation and testing files.

    Parameters
    ----------
    images_dir : str
        the directory with the .jpg files
    train_videos : list
        the list of training videos
    val_videos : list
        the list of validation videos
    test_videos : list
        the list of testing videos
    
    Returns
    -------
    tuple
        the list of training, validation  and testing images
    """
    # Initialize empty lists to store image names for train and validation sets
    train_images, val_images, test_images = [], [], []

    # Iterate through all images in the directory
    for image_file in images_dir.rglob("*.jpg"):
        video_name_part = "_".join(image_file.stem.split("_")[:-1])
        
        if video_name_part in train_videos:
            train_images.append(image_file.name)
        elif video_name_part in val_videos:
            val_images.append(image_file.name)
        elif video_name_part in test_videos:
            test_images.append(image_file.name)
    
    logging.info(f"Split completed: Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
    return train_images, val_images, test_images

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
        the list of image files to move
    dest_dir_images : Path
        the destination directory for the image files
    dest_dir_labels : Path
        the destination directory for the label files
    """        
    logging.info(f"Copying YOLO files to {dest_dir_images} and {dest_dir_labels}")
    # Move the files to the new directory
    for file_path in files_to_move_lst:
        file_path = Path(file_path)
        folder_name = str(file_path)[:-11]
        # Construct the full source paths for the image and label
        src_image_path = DetectionPaths.images_input_dir / folder_name / file_path
        src_label_path = YoloPaths.face_labels_input_dir / file_path.with_suffix('.txt')
        # Copy the images and move the label to their new destinations
        shutil.copy(src_image_path, dest_dir_images)
        shutil.copy(src_label_path, dest_dir_labels)
    logging.info("YOLO files copied successfully")


def prepare_yolo_dataset(
    destination_dir: Path,
    train_files: list,
    val_files: list,
    test_files: list
):
    """
    This function moves the training, validation and testing files to the new yolo directories.

    Parameters
    ----------
    destination_dir : Path
        the destination directory to store the training, validation and testing sets
    train_files : list
        the list of image training files
    val_files : list
        the list of image validation files
    test_files : list
        the list of image testing files
    """
    # Define source directory and new directories for training
    train_dir_images = destination_dir / "images/train"
    val_dir_images = destination_dir / "images/val"
    test_dir_images = destination_dir / "images/test"
    train_dir_labels = destination_dir / "labels/train"
    val_dir_labels = destination_dir / "labels/val"
    test_dir_labels = destination_dir / "labels/test"
    
    # Create necessary directories if they don't exist
    for path in [train_dir_images, val_dir_images,  test_dir_images, train_dir_labels, val_dir_labels, test_dir_labels]:
        path.mkdir(parents=True, exist_ok=True)
    
    # Move the files to the new directories
    copy_yolo_files(train_files, train_dir_images, train_dir_labels)
    copy_yolo_files(val_files, val_dir_images, val_dir_labels)  
    copy_yolo_files(test_files, test_dir_images, test_dir_labels)

def copy_mtcnn_files(
    train_files: list,
    val_files: list, 
    test_files: list,
    train_dir_images: Path,
    val_dir_images: Path,
    test_dir_images: Path,
    train_labels_path: Path,
    val_labels_path: Path,
    test_labels_path: Path
    )-> None:
    """
    This function moves the training, validation and testing files to the new mtcnn directories.

    Parameters
    ----------
    train_files : list
        the list of training files
    val_files : list
        the list of validation files
    test_files : list
        the list of testing files
    train_dir_images: Path
        the directory to store the training images
    val_dir_images: Path
        the directory to store the validation images
    test_dir_images: Path
        the directory to store the testing images
    train_labels_path: Path
        the path to store the training labels
    val_labels_path: Path
        the path to store the validation labels
    test_labels_path: Path
        the path to store the testing labels
    """
    logging.info(f"Moving MTCNN files to {train_dir_images}, {val_dir_images}, {test_dir_images}, {train_labels_path}, {val_labels_path} and {test_labels_path}")
    # Move the images to the training and validation directories
    # Convert all elements in train_files and val_files to Path objects
    train_files = [Path(file_path) for file_path in train_files]
    val_files = [Path(file_path) for file_path in val_files]
    test_files = [Path(file_path) for file_path in test_files]
    
    for file_path_train in train_files:
        src_image_path = DetectionPaths.images_input / file_path_train.name
        dest_image_path = train_dir_images / file_path_train.name
        shutil.copy(src_image_path, dest_image_path)
    for file_path_val in val_files:
        src_image_path = DetectionPaths.images_input / file_path_val.name
        dest_image_path = val_dir_images / file_path_val.name
        shutil.copy(src_image_path, dest_image_path)
    for file_path_test in test_files:
        src_image_path = DetectionPaths.images_input / file_path_test.name
        dest_image_path = test_dir_images / file_path_test.name
        shutil.copy(src_image_path, dest_image_path)
    
    # Convert lists to sets and extract the file names
    train_set = set(train_files)
    val_set = set(val_files)
    test_set = set(test_files)
    file_names_in_train_set = {path.name.rsplit('.', 1)[0] for path in train_set}
    file_names_in_val_set = {path.name.rsplit('.', 1)[0] for path in val_set}
    file_names_in_test_set = {path.name.rsplit('.', 1)[0] for path in test_set}

    # Move the labels to the training and validation directories
    with MtcnnPaths.labels_file_path.open('r') as original_file, train_labels_path.open('w') as train_file, val_labels_path.open('w') as validation_file, test_labels_path.open('w') as test_file:
        for line in original_file:
            image_file_name = line.split()[0]  # Assuming the file name is the first element
            if image_file_name in file_names_in_train_set:
                train_file.write(line)
            elif image_file_name in file_names_in_val_set:
                validation_file.write(line)
            elif image_file_name in file_names_in_test_set:
                test_file.write(line)
    logging.info("MTCNN files copied successfully")


def prepare_mtcnn_dataset(
    destination_dir: Path,
    train_files: list,
    val_files: list,
    test_files: list
):
    """
    This function moves the training, validation and testing files to the new yolo directories.

    Parameters
    ----------
    destination_dir : Path
        the destination directory to store the training and validation sets
    train_files : list
        the list of training files
    val_files : list
        the list of validation files
    test_files : list

    """
    # Define source directory and new directories for training
    train_dir_images = destination_dir / "train/images"
    train_labels_path = destination_dir / "train/train.txt"
    val_dir_images = destination_dir / "val/images"
    val_labels_path = destination_dir / "val/val.txt"
    test_dir_images = destination_dir / "test/images"
    test_labels_path = destination_dir / "test/test.txt"

    # Create necessary directories if they don't exist
    for path in [train_dir_images, val_dir_images, train_labels_path.parent, val_labels_path.parent, test_dir_images, test_labels_path.parent]:
        path.mkdir(parents=True, exist_ok=True)
    
    # Move the files to the new directories
    copy_mtcnn_files(train_files, 
                    val_files, 
                    test_files,
                    train_dir_images, 
                    val_dir_images,
                    test_dir_images,
                    train_labels_path, 
                    val_labels_path,
                    test_labels_path,
    )

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

def prepare_dataset(
    model_target: str, 
    yolo_target: str = None,
    destination_dir: Path = None,
    train_files: list = None,
    val_files: list = None,
    test_files: list = None
):
    """
    Prepares the dataset for the specified model (mtcnn or yolo).
    
    Parameters
    ----------
    model_target : str
        the model to prepare the dataset for
    yolo_target : str
        the target for the YOLO model (person or face)
    destination_dir : Path
        the destination directory to store the training and validation sets
    train_files : list
        the list of training files
    val_files : list
        the list of validation files
    test_files : list
        the list of testing files
    """
    if mode == "yolo":
        if target not in ["person", "face"]:
            raise ValueError("YOLO target must be 'person' or 'face'")
        target_dir = YoloPaths.person_data_input_dir if target == "person" else YoloPaths.face_data_input_dir
        prepare_yolo_dataset(target_dir, train_files, val_files, test_files)
    elif mode == "mtcnn":
        prepare_mtcnn_dataset(MtcnnPaths.data_input, train_files, val_files, test_files)
    else:
        raise ValueError("Invalid mode. Use 'mtcnn' or 'yolo'")
    
def get_video_lengths() -> list:
    """Returns list of (video_name, length) tuples for videos with annotations"""
    video_lengths = []
    
    # Connect to annotations database
    conn = sqlite3.connect(DetectionPaths.annotations_db_path)
    cursor = conn.cursor()
    
    # Get unique video IDs from annotations and join with videos table
    cursor.execute("""
        SELECT DISTINCT v.file_name 
        FROM annotations a
        JOIN videos v ON a.video_id = v.id
        WHERE v.file_name NOT LIKE '%id255237_2022_05_08_04%'
    """)
    video_files_list = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    # Get length for each video
    for video in video_files_list:
        video_name = os.path.splitext(video)[0]
        possible_files = list(DetectionPaths.quantex_videos_input_dir.glob(f"{video_name}.mp4")) + \
                        list(DetectionPaths.quantex_videos_input_dir.glob(f"{video_name}.MP4"))
        
        if not possible_files:
            logging.warning(f"Video file not found: {video_name}")
            continue
            
        video_path = possible_files[0]
        length = get_video_length(video_path)
        video_lengths.append((video_name, length))
    
    logging.info(f"Found {len(video_lengths)} videos with annotations")
    return video_lengths


def balanced_train_val_test_split(
    train_ratio: float
) -> tuple:
    
    """
    This function splits the dataset into training, validation and testing sets
    while balancing the cumulative length of the videos in each set.
    
    Parameters
    ----------
    train_ratio : float
        the ratio to split the dataset
        
    Returns
    -------
    tuple
        the list of training, validation and testing videos
    """
    video_files_with_length = get_video_lengths()

    # Set the random seed for reproducibility
    random.seed(TrainingConfig.random_seed)
    # Shuffle the list to randomize the selection
    random.shuffle(video_files_with_length)
    
    # Calculate split sizes
    total_videos = len(video_files_with_length)
    num_train_videos = int(train_ratio * total_videos)
    num_val_videos = (total_videos - num_train_videos) // 2
    num_test_videos = total_videos - num_train_videos - num_val_videos
    
    train_videos, val_videos, test_videos = [], [], []
    train_length, val_length, test_length = 0, 0, 0
    
    for video in video_files_with_length:
        if len(train_videos) < num_train_videos:
            train_videos.append(video[0])
            train_length += video[1]
        elif len(val_videos) < num_val_videos:
            val_videos.append(video[0])
            val_length += video[1]
        else:
            test_videos.append(video[0])
            test_length += video[1]
    
    logging.info(f"Train set: {len(train_videos)} videos")
    logging.info(f"Val set: {len(val_videos)} videos")
    logging.info(f"Test set: {len(test_videos)} videos")
    
    return train_videos, val_videos, test_videos

def main(model: str, yolo_target: str) -> None:
    """ 
    This function prepares the training dataset for the specified model and target.
    
    Parameters
    ----------
    model : str
        the model to prepare the dataset for
    yolo_target : str
        the target for the YOLO model (person or face)    
    """
    # Environment setup
    os.environ['OMP_NUM_THREADS'] = '10'
    if args.model_target == "yolo" and args.yolo_target == "person":
        create_missing_annotation_files(DetectionPaths.images_input_dir, YoloPaths.person_labels_input_dir)   
    elif args.model_target == "yolo" and args.yolo_target == "face":
        create_missing_annotation_files(DetectionPaths.images_input_dir, YoloPaths.face_labels_input_dir)

    # Split video files into training, validation, and testing sets
    train_videos, val_videos, test_videos = balanced_train_val_test_split(TrainingConfig.train_test_split_ratio)
    train_files, val_files, test_files = images_train_val_test_split(
        DetectionPaths.images_input_dir, 
        train_videos, 
        val_videos,
        test_videos,
    )

    # Prepare the dataset based on the mode and target
    prepare_dataset(
        mode=args.mode, 
        target=args.yolo_target, 
        train_files=train_files, 
        val_files=val_files, 
        test_files=test_files
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training datasets")
    parser.add_argument("--model_target", required=True, choices=["mtcnn", "yolo"], help="Preparation model: mtcnn or yolo")
    parser.add_argument("--yolo_target", choices=["person", "face"], help="Target for YOLO: person or face")
    
    args = parser.parse_args()
    main(model=args.model_target, yolo_target=args.yolo_target)
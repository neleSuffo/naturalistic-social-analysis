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

def get_class_to_total_ratio(annotation_folder: Path, image_folder: Path) -> float:
    """
    This function calculates the ratio of images with classes to total images.
    
    Parameters
    ----------
    annotation_folder : str
        the directory with the annotation files
    image_folder : str
        the directory with the image files
        
    Returns
    -------
    class_to_total_ratio: float
        the ratio of images with classes to total images
    total_images: int
        the list of total images from annotated videos
    
    """
    # Step 1: Extract unique video names from annotation files
    video_names = set()
    for annotation_file in os.listdir(annotation_folder):
        if annotation_file.endswith('.txt'):
            parts = annotation_file.split('_')
            video_name = "_".join(parts[:8])
            video_names.add(video_name)
    logging.info(f"Found {len(video_names)} unique video names")
    
    # Step 2: Count total images and images with faces
    total_images = []
    total_images_count = 0
    images_with_class_count = 0
    for video_name in video_names:
        video_path = image_folder / video_name
        if os.path.isdir(video_path):
            total_images_count += len(os.listdir(video_path))
            total_images.extend(os.listdir(video_path))
    images_with_class_count += len(os.listdir(annotation_folder))
    
    logging.info(f"Total images count: {total_images_count}, Images with class count: {images_with_class_count}")

    # Step 3: Calculate the ratio
    class_to_total_ratio = images_with_class_count / total_images_count if total_images_count > 0 else 0
    logging.info(f"Class to total ratio: {class_to_total_ratio}")
    return class_to_total_ratio, total_images

def images_train_val_test_split(
    image_folder: str,
    images_list: list,
    train_ratio: float,
    class_to_total_ratio: float
) -> tuple:
    """
    This function shuffles all images and splits them into training, validation, and testing sets.

    Parameters
    ----------
    images_list : list
        the list of images to split
    train_ratio : float
        the ratio of images to use for training
    class_to_total_ratio : float
        the ratio of images with classes to total images
    
    Returns
    -------
    tuple
        the list of training, validation, and testing images
    """
    val_ratio = (1 - train_ratio) / 2
    test_ratio = 1 - train_ratio - val_ratio

    # Ensure the sum of the ratios equals 1
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "The sum of ratios must be 1."
    
    train_images = []
    val_images = []
    test_images = []

    # Shuffle the images
    random.seed(TrainingConfig.random_seed)
    random.shuffle(images_list)
    
    # Split the images into training, validation, and testing sets
    total_images = len(images_list)
    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)
    test_size = total_images - (train_size + val_size)
    
    # Split the images into training, validation, and testing sets
    train_images = images_list[:train_size]
    val_images = images_list[train_size:train_size + val_size]
    test_images = images_list[train_size + val_size:]

    logging.info(f"Train set: {len(train_images)} frames")
    logging.info(f"Val set: {len(val_images)} frames")
    logging.info(f"Test set: {len(test_images)} videos")

    return train_images, val_images, test_images

def main(model_target: str, yolo_target: str) -> None:
    """
    Main function to prepare the training dataset for the specified model and target.

    Parameters
    ----------
    model_target : str
        The model to prepare the dataset for ('yolo' or 'mtcnn').
    yolo_target : str
        The target for the YOLO model ('person' or 'face').
    """
    # Environment setup
    os.environ['OMP_NUM_THREADS'] = '10'
    
    # Create missing annotation files for YOLO
    if model_target == "yolo":
        if yolo_target == "person":
            class_to_total_ratio, total_images = get_class_to_total_ratio(YoloPaths.person_labels_input_dir, DetectionPaths.images_input_dir)
            create_missing_annotation_files(DetectionPaths.images_input_dir, YoloPaths.person_labels_input_dir)
        elif yolo_target == "face":
            class_to_total_ratio, total_images = get_class_to_total_ratio(YoloPaths.face_labels_input_dir, DetectionPaths.images_input_dir)
            create_missing_annotation_files(DetectionPaths.images_input_dir, YoloPaths.face_labels_input_dir)
    
    # Split video files into training, validation, and testing sets
    train_images, val_images, test_images = images_train_val_test_split(DetectionPaths.images_input_dir, total_images, TrainingConfig.train_test_split_ratio, class_to_total_ratio)
    
    # Prepare dataset based on the model target
    if model_target == "yolo":
        prepare_yolo_dataset(
            YoloPaths.face_data_input_dir if yolo_target == "face" else YoloPaths.person_data_input_dir,
            train_images, val_images, test_images
        )
    elif model_target == "mtcnn":
        prepare_mtcnn_dataset(
            MtcnnPaths.data_input, train_images, val_images, test_images
        )
    else:
        raise ValueError(f"Unsupported model target: {model_target}")

    logging.info("Dataset preparation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare datasets for training.")
    parser.add_argument("model_target", type=str, choices=["yolo", "mtcnn"], help="Target model for preparation.")
    parser.add_argument("--yolo_target", type=str, choices=["person", "face"], help="YOLO target type (person or face).")
    args = parser.parse_args()
    
    main(args.model_target, args.yolo_target)
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

def split_yolo_data(total_images: list, 
                    output_folder: Path, 
                    class_to_total_ratio: float, 
                    train_test_split_ratio: float=TrainingConfig.train_test_split_ratio):
    """
    Splits data into train, validation, and test sets while ensuring that 
    missing annotation files are created only in the target directory.

    Parameters
    ----------
    total_images : list
        The list of total images from annotated videos.
    output_folder : Path
        The directory to store the training, validation, and testing sets.
    class_to_total_ratio : float
        The ratio of images with classes to total images.
    train_test_split_ratio : float
        The ratio of images to use for training, default is 0.8.
    """

    # Define split ratios
    train_ratio = train_test_split_ratio  # e.g., 0.8 for training
    val_ratio = (1 - train_ratio) / 2  # Split remaining equally for val & test
    test_ratio = (1 - train_ratio) / 2

    # Shuffle data for randomness
    random.shuffle(total_images)

    # Compute split indices
    total_files = len(total_images)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)

    # Split into train, val, and test sets
    train_files = total_images[:train_count]
    val_files = total_images[train_count:train_count + val_count]
    test_files = total_images[train_count + val_count:]

    # Define subsets
    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    # Ensure output directories exist
    for split in splits.keys():
        (output_folder / split / "images").mkdir(parents=True, exist_ok=True)
        (output_folder / split / "annotations").mkdir(parents=True, exist_ok=True)

    # Copy images & check annotation files
    for split, files in splits.items():
        for image_file in files:
            image_file_path = Path(image_file)
            folder_name = image_file_path.stem[:-11]  # Extract video folder

            # Source paths
            src_image_path = DetectionPaths.images_input_dir / folder_name / image_file_path.name
            src_label_path = YoloPaths.face_labels_input_dir / image_file_path.with_suffix('.txt').name

            # Target paths
            image_dst = output_folder / split / "images" / image_file_path.name
            annotation_dst = output_folder / split / "annotations" / image_file_path.with_suffix('.txt').name

            # Copy image
            shutil.copy(src_image_path, image_dst)

            # Check if annotation file exists; copy or create it
            if src_label_path.exists():
                shutil.copy(src_label_path, annotation_dst)
            else:
                annotation_dst.touch()  # Create an empty annotation file

    logging.info(f"Data successfully split into train ({train_ratio:.2f}), val ({val_ratio:.2f}), test ({test_ratio:.2f}).") 
        
        
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
            train_images, val_images, test_images = split_yolo_data(DetectionPaths.images_input_dir, 
                                                                    total_images, 
                                                                    YoloPaths.person_labels_input_dir, 
                                                                    YoloPaths.person_data_input_dir,
                                                                    class_to_total_ratio,
                                                                    TrainingConfig.train_test_split_ratio)

        elif yolo_target == "face":
            class_to_total_ratio, total_images = get_class_to_total_ratio(YoloPaths.face_labels_input_dir, DetectionPaths.images_input_dir)
            train_images, val_images, test_images = split_yolo_data(DetectionPaths.images_input_dir, 
                                                                    total_images, 
                                                                    YoloPaths.face_labels_input_dir, 
                                                                    YoloPaths.face_data_input_dir,
                                                                    class_to_total_ratio,
                                                                    TrainingConfig.train_test_split_ratio)
        
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
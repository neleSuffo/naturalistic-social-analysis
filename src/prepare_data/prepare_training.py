import shutil
import random
import logging
import os
import sqlite3
import subprocess
from pathlib import Path
from constants import DetectionPaths, YoloPaths
from config import TrainingConfig
import argparse
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def balance_dataset(model_target: str, yolo_target: str):
    """
    Balances a dataset by randomly selecting an equal number of images with and without faces.

    Parameters:
    ----------
    model_target : str
        The target model for preparation (e.g., "yolo").
    yolo_target : str
        The target type for YOLO (e.g., "person" or "face").
    """
    # Define paths based on model and YOLO target
    paths = {
        ("yolo", "person"): YoloPaths.person_data_input_dir,
        ("yolo", "face"): YoloPaths.face_data_input_dir,
        ("yolo", "gaze"): YoloPaths.gaze_data_input_dir
    }
    
    data_input_dir = paths.get((model_target, yolo_target))
    
    if data_input_dir is None:
        logging.error(f"Invalid combination of model_target='{model_target}' and yolo_target='{yolo_target}'.")
        return
    
    annotation_dir = data_input_dir / "labels/train"
    images_dir = data_input_dir / "images/train"
    balanced_annotations_dir = data_input_dir / "labels/train_balanced"
    balanced_images_dir = data_input_dir / "images/train_balanced"

    logging.info(f"Balancing dataset in {images_dir} and {annotation_dir}...")

    # Ensure directories exist
    if not annotation_dir.exists() or not images_dir.exists():
        logging.error(f"Missing input directories: {annotation_dir} or {images_dir}")
        return
    
    images_with_class = []
    images_without_class = []

    # Iterate over annotation files to classify images
    for annotation_path in annotation_dir.iterdir():
        if annotation_path.suffix == ".txt":
            image_path = images_dir / f"{annotation_path.stem}.jpg"
            
            # Check if the annotation file is not empty
            if yolo_target == "face":
                if annotation_path.stat().st_size > 0:
                    images_with_class.append((image_path, annotation_path))
                else:
                    images_without_class.append((image_path, annotation_path))
            elif yolo_target == "person":
                with open(annotation_path, 'r') as file:
                    lines = file.readlines()
                    # Check if any annotation has class 0
                    has_class_0 = any(line.strip().split()[0] == "0" for line in lines)
                    if has_class_0:
                        images_with_class.append((image_path, annotation_path))
                    else:
                        images_without_class.append((image_path, annotation_path))

    num_img_with_classes = len(images_with_class)
    
    if num_img_with_classes == 0:
        logging.warning(f"No images with {yolo_target}s found. Skipping dataset balancing.")
        return

    logging.info(f"Found {num_img_with_classes} images with {yolo_target}.")

    if len(images_without_class) < num_img_with_classes:
        logging.warning(f"Not enough images without {yolo_target}s to balance the dataset. Using all available.")

    # Randomly select an equal number of images without class (person or face)
    images_without_class_sample = random.sample(images_without_class, min(num_img_with_classes, len(images_without_class)))

    # Combine the lists to form the balanced dataset
    balanced_dataset = images_with_class + images_without_class_sample

    # Ensure balanced dataset directories exist
    balanced_images_dir.mkdir(parents=True, exist_ok=True)
    balanced_annotations_dir.mkdir(parents=True, exist_ok=True)

    # Copy the selected files to the balanced dataset directories
    for image_path, annotation_path in balanced_dataset:
        shutil.copy(image_path, balanced_images_dir / image_path.name)
        shutil.copy(annotation_path, balanced_annotations_dir / annotation_path.name)

    logging.info(f"Balanced dataset created with {len(images_with_class)} images with {yolo_target}s and {len(images_without_class_sample)} images without {yolo_target}s.")
    
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
def get_total_number_of_annotated_frames(label_path: Path, image_folder: Path) -> list:
    """
    This function returns the total number of annotated frames in the dataset.
    
    Parameters
    ----------
    label_path : Path
        the path to the label files
    image_folder : Path
        the path to the image folder
        
    Returns
    -------
    list
        the total number of images
    
    """
    video_names = set()
    total_images = []
    # Step 1: Get unique video names
    for annotation_file in os.listdir(label_path):
        if annotation_file.endswith('.txt'):
            parts = annotation_file.split('_')
            video_name = "_".join(parts[:8])
            video_names.add(video_name)
    logging.info(f"Found {len(video_names)} unique video names")
    
    # Step 2: Count total images
    for video_name in video_names:
        video_path = image_folder / video_name
        if os.path.isdir(video_path):
            total_images.extend(os.listdir(video_path)) 
    return total_images    

def get_class_distribution(total_images: list, annotation_folder: Path, yolo_target: str):
    """
    Categorizes images based on the presence of specific class annotations and computes the class-to-total ratio.

    Parameters
    ----------
    total_images : list
        List of all image file paths.
    annotation_folder : Path
        Path to the directory containing annotation files.
    yolo_target : str
        Target type ("gaze", "face", or "person") to determine counting logic.

    Returns
    -------
    tuple : (list, list, float)
        - List of images with class annotations.
        - List of images without class annotations.
        - Class-to-total ratio.
    """
    with_class, without_class = [], []

    for image_file in total_images:
        annotation_file = annotation_folder / Path(image_file).with_suffix('.txt').name

        if annotation_file.exists() and annotation_file.stat().st_size > 0:
            with open(annotation_file, 'r') as f:
                lines = f.readlines()
                if yolo_target == "gaze":
                    # Check every line in the annotation file
                    for i, line in enumerate(lines):
                        class_id = line.strip().split()[0]
                        if class_id == "0":  # 0 = no gaze
                            without_class.append(f"{Path(image_file).stem}_face_{i}.jpg")
                        elif class_id == "1":  # 1 = gaze
                            with_class.append(f"{Path(image_file).stem}_face_{i}.jpg")
                else:
                    has_class_0 = any(line.strip().split()[0] == "0" for line in lines)
                    if has_class_0:
                        with_class.append(image_file)
                    else:
                        without_class.append(image_file)
    if yolo_target == "gaze":
        # Calculate class-to-no-class ratio (number of images with gaze directed at child vs. not)
        total_gaze = len(with_class) + len(without_class)
        class_to_no_class_ratio = len(with_class) / total_gaze if total_gaze > 0 else 0
        logging.info(f"Class-to-No-Class Ratio: {class_to_no_class_ratio:.4f} ({len(with_class)}/{len(without_class)})")
        return with_class, without_class, class_to_no_class_ratio
    
    class_to_total_ratio = len(with_class) / len(total_images) if total_images else 0
    logging.info(f"Class-to-Total Ratio: {class_to_total_ratio:.4f} ({len(with_class)}/{len(total_images)})")
    return with_class, without_class, class_to_total_ratio

def stratified_split(data, train_ratio, val_ratio):
    """
    Splits data into train, validation, and test sets maintaining proportions.

    Parameters
    ----------
    data : list
        List of image file paths.
    train_ratio : float
        Proportion of data for training.
    val_ratio : float
        Proportion of data for validation.

    Returns
    -------
    tuple : (list, list, list)
        - Train split
        - Validation split
        - Test split
    """
    total = len(data)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    return data[:train_count], data[train_count:train_count + val_count], data[train_count + val_count:]

def copy_files_to_split(split_name, files, annotation_folder, output_folder, yolo_target):
    """
    Copies image and annotation files into their respective split directories.

    Parameters
    ----------
    split_name : str
        The split name ('train', 'val', 'test').
    files : list
        List of image file paths.
    annotation_folder : Path
        Path to the annotation directory.
    output_folder : Path
        Path to the output directory.
    yolo_target : str
        Target type ("face" or "gaze")
    """
    for image_file in files:
        image_file_path = Path(image_file)
        folder_name = "_".join(image_file_path.stem.split("_")[:8])

        # Define source paths
        if yolo_target == "gaze":
            src_image_path = DetectionPaths.gaze_images_input_dir / image_file_path.name
        else:
            src_image_path = DetectionPaths.images_input_dir / folder_name / image_file_path.name
        
        # Define target paths
        image_dst = output_folder / "images" / split_name / image_file_path.name
        annotation_dst = output_folder / "labels" / split_name / image_file_path.with_suffix('.txt').name

        # Copy image
        if src_image_path.exists():
            shutil.copy(src_image_path, image_dst)
        else:
            logging.warning(f"Skipping {src_image_path} - source file not found")
            continue
        
        if yolo_target == "gaze":
            # Extract base name and face index from filename (e.g., video_name_face_1)
            base_name = "_".join(image_file_path.stem.split("_")[:9])  # Remove '_face_X'
            face_idx = int(image_file_path.stem.split("_")[-1])  # Get X from '_face_X'
            
            # Source annotation file contains all faces
            src_label_path = annotation_folder / f"{base_name}.txt"
            
            if src_label_path.exists():
                with open(src_label_path, 'r') as f:
                    lines = f.readlines()
                    if face_idx < len(lines):
                        # Create annotation file with only the relevant line
                        with open(annotation_dst, 'w') as out_f:
                            out_f.write(lines[face_idx])
                    else:
                        logging.warning(f"Face index {face_idx} not found in {src_label_path}")
                        annotation_dst.touch()
            else:
                annotation_dst.touch()  # Create empty file if source doesn't exist
        else:
            # Original behavior for face detection
            src_label_path = annotation_folder / image_file_path.with_suffix('.txt').name
            if src_label_path.exists():
                shutil.copy(src_label_path, annotation_dst)
            else:
                annotation_dst.touch()

    logging.info(f"Copied {len(files)} images to {split_name} split.")
    
def split_yolo_data(total_images: list, 
                    annotation_folder: Path,
                    output_folder: Path, 
                    yolo_target: str,
                    train_test_split_ratio: float = TrainingConfig.train_test_split_ratio):
    """
    Splits YOLO image dataset into train, validation, and test sets while ensuring
    stratification based on the class-to-total ratio.

    Parameters
    ----------
    total_images : list
        List of image file paths.
    annotation_folder : Path
        Path to annotation files.
    output_folder : Path
        Path to store the output splits.
    yolo_target : str
        Target type ("face", "person", or "gaze").
    train_test_split_ratio : float
        Ratio of images to use for training, default is 0.8.
    """

    # Define split ratios
    train_ratio = train_test_split_ratio  # e.g., 0.8 for training
    val_ratio = (1 - train_ratio) / 2  # Split remaining equally for val & test

    # Ensure output directories exist
    for split in ["train", "val", "test"]:
        (output_folder / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_folder / "labels"/ split).mkdir(parents=True, exist_ok=True)

    # Get class distribution
    with_class, without_class, class_to_total_ratio = get_class_distribution(total_images, annotation_folder, yolo_target)

    # Shuffle for randomness
    random.shuffle(with_class)
    random.shuffle(without_class)

    # Perform stratified splitting
    train_with, val_with, test_with = stratified_split(with_class, train_ratio, val_ratio)
    train_without, val_without, test_without = stratified_split(without_class, train_ratio, val_ratio)

    # Merge class and no-class images into final splits
    splits = {
        "train": train_with + train_without,
        "val": val_with + val_without,
        "test": test_with + test_without
    }

    logging.info(f"Train split: {len(splits['train'])} images ({len(train_with)} class, {len(train_without)} no-class).")
    logging.info(f"Val split: {len(splits['val'])} images ({len(val_with)} class, {len(val_without)} no-class).")
    logging.info(f"Test split: {len(splits['test'])} images ({len(test_with)} class, {len(test_without)} no-class).")

    # Shuffle each split to mix class and no-class images
    for split in splits:
        random.shuffle(splits[split])

    logging.info("Copying images and annotations to output folder...")
    
    # Copy files into respective split directories
    for split, files in splits.items():
        copy_files_to_split(split, files, annotation_folder, output_folder, yolo_target)

    logging.info(f"Data successfully split into train ({train_ratio:.2f}), val ({val_ratio:.2f}), test ({val_ratio:.2f}).")
        
def main(model_target: str, yolo_target: str):
    """ 
    This function prepares the dataset for training the model.
    
    Parameters
    ----------
    model_target : str
        the target model for preparation
    yolo_target : str
        the target type for YOLO (person or face)

    Returns
    -------
    None
    
    """
    logging.info(f"Starting dataset preparation for {model_target}...")
    os.environ['OMP_NUM_THREADS'] = '10'
    
    if model_target == "yolo":
        label_path = YoloPaths.person_labels_input_dir if yolo_target == "person" else YoloPaths.face_labels_input_dir if yolo_target == "face" else YoloPaths.gaze_labels_input_dir
        data_path = YoloPaths.person_data_input_dir if yolo_target == "person" else YoloPaths.face_data_input_dir if yolo_target == "face" else YoloPaths.gaze_data_input_dir        
        
        logging.info(f"Processing YOLO dataset for target: {yolo_target}")
        total_images = get_total_number_of_annotated_frames(label_path, DetectionPaths.images_input_dir)
        split_yolo_data(total_images, label_path, data_path, yolo_target, TrainingConfig.train_test_split_ratio)
    elif model_target == "mtcnn":
        prepare_mtcnn_dataset(MtcnnPaths.data_input, train_images, val_images, test_images)
    else:
        logging.error(f"Unsupported model target: {model_target}")
        raise ValueError(f"Unsupported model target: {model_target}")
    
    logging.info("Dataset preparation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare datasets for training.")
    parser.add_argument("model_target", type=str, choices=["yolo", "mtcnn"], help="Target model for preparation.")
    parser.add_argument("--yolo_target", type=str, choices=["person", "face", "gaze"], help="YOLO target type (person or face).")
    args = parser.parse_args()
    
    main(args.model_target, args.yolo_target)
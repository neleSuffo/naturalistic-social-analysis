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
    if yolo_target == "person+face":
        with_class = set()
        without_class = set()
        person_images = set()
        face_images = set()
        child_images = set()

        for image_file in total_images:
            annotation_file = annotation_folder / Path(image_file).with_suffix('.txt').name
            if annotation_file.exists() and annotation_file.stat().st_size > 0:
                with open(annotation_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) == 0:
                        without_class.add(image_file)
                    else:
                        with_class.add(image_file)
                        for line in lines:
                            parts = line.strip().split()
                            if parts:
                                class_id = parts[0]
                                if class_id == "0":  # 0 = person
                                    person_images.add(image_file)
                                elif class_id == "1":  # 1 = face
                                    face_images.add(image_file)
                                elif class_id == "2":  # 2 = child body parts
                                    child_images.add(image_file)
            else:
                without_class.append(image_file)
        total_images_count = len(total_images)
        person_count = len(person_images)
        face_count = len(face_images)
        child_count = len(child_images)
        
        person_ratio = person_count / total_images_count if total_images_count else 0
        face_ratio = face_count / total_images_count if total_images_count else 0
        child_ratio = child_count / total_images_count if total_images_count else 0
        overall_class_ratio = len(with_class) / total_images_count if total_images_count else 0

        logging.info(f"Person Class Ratio: {person_ratio:.4f} ({person_count}/{total_images_count})")
        logging.info(f"Face Class Ratio: {face_ratio:.4f} ({face_count}/{total_images_count})")
        logging.info(f"Child Body Part Ratio: {child_ratio:.4f} ({child_count}/{total_images_count})")
        logging.info(f"Overall Class-to-Total Ratio: {overall_class_ratio:.4f} ({len(with_class)}/{total_images_count})")
        
        return with_class, without_class, overall_class_ratio
                                
    elif yolo_target == "gaze":
        with_class = []
        without_class = []
        for image_file in total_images:
            annotation_file = annotation_folder / Path(image_file).with_suffix('.txt').name
            if annotation_file.exists() and annotation_file.stat().st_size > 0:
                with open(annotation_file, 'r') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        class_id = line.strip().split()[0]
                        if class_id == "0":  # 0 = no gaze
                            without_class.append(f"{Path(image_file).stem}_face_{i}.jpg")
                        elif class_id == "1":  # 1 = gaze
                            with_class.append(f"{Path(image_file).stem}_face_{i}.jpg")
            else:
                without_class.append(image_file)
        total_gaze = len(with_class) + len(without_class)
        ratio = len(with_class) / total_gaze if total_gaze > 0 else 0
        logging.info(f"Class-to-No-Class Ratio: {ratio:.4f} ({len(with_class)}/{len(without_class)})")
        return with_class, without_class, ratio
    
    else:
        # Default logic for targets like "person" or "face"
        with_class = []
        without_class = []
        for image_file in total_images:
            annotation_file = annotation_folder / Path(image_file).with_suffix('.txt').name
            if annotation_file.exists() and annotation_file.stat().st_size > 0:
                with open(annotation_file, 'r') as f:
                    lines = f.readlines()
                    if any(line.strip().split()[0] == "0" for line in lines):
                        with_class.append(image_file)
                    else:
                        without_class.append(image_file)
            else:
                without_class.append(image_file)
        total_images_count = len(total_images)
        overall_ratio = len(with_class) / total_images_count if total_images_count else 0
        logging.info(f"Class-to-Total Ratio: {overall_ratio:.4f} ({len(with_class)}/{total_images_count})")
        return with_class, without_class, overall_ratio

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

def get_labels_for_image(image_file: str, annotation_folder: Path) -> tuple:
    """
    Determines the presence (1) or absence (0) of person and face in an image based on its annotation file.
    
    Parameters:
    ----------
    image_file : str
        The image file name (or path) as a string.
    annotation_folder : Path
        Directory where annotation files are stored.

    Returns:
    -------
    tuple
        A tuple (person_flag, face_flag) where each flag is 1 if present, else 0.
    """
    annotation_file = annotation_folder / (Path(image_file).with_suffix('.txt').name)
    person_flag = 0
    face_flag = 0

    if annotation_file.exists() and annotation_file.stat().st_size > 0:
        with annotation_file.open('r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                # Assuming class "0" indicates person and class "1" indicates face
                if parts[0] == "0":
                    person_flag = 1
                if parts[0] == "1":
                    face_flag = 1
    return (person_flag, face_flag)

def stratified_split_person_face(images: list, annotation_folder: Path, train_ratio: float, val_ratio: float):
    """
    Ensures that the ratio of person to total and face to total is approximately preserved
    in training, validation, and test sets, counting each image only once per class.
    """
    # Assign each image a (person_flag, face_flag) label
    label_groups = defaultdict(list)
    for img in images:
        label = get_labels_for_image(img, annotation_folder)
        label_groups[label].append(img)

    # Split each group while maintaining overall ratios
    train_split, val_split, test_split = [], [], []
    
    for label, imgs in label_groups.items():
        random.shuffle(imgs)
        total = len(imgs)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)

        train_split.extend(imgs[:train_count])
        val_split.extend(imgs[train_count:train_count + val_count])
        test_split.extend(imgs[train_count + val_count:])

    # Function to compute person/face ratios while counting each image only once per category
    def compute_ratios(split):
        person_images = set()
        face_images = set()

        for img in split:
            person_flag, face_flag = get_labels_for_image(img, annotation_folder)
            if person_flag:
                person_images.add(img)  # Count each image only once
            if face_flag:
                face_images.add(img)

        person_ratio = len(person_images) / len(split) if split else 0
        face_ratio = len(face_images) / len(split) if split else 0
        return person_ratio, face_ratio

    # Compute per-split person/face ratios
    train_ratios = compute_ratios(train_split)
    val_ratios = compute_ratios(val_split)
    test_ratios = compute_ratios(test_split)

    logging.info(f"Train: {len(train_split)} images, Person ratio: {train_ratios[0]:.2f}, Face ratio: {train_ratios[1]:.2f}")
    logging.info(f"Validation: {len(val_split)} images, Person ratio: {val_ratios[0]:.2f}, Face ratio: {val_ratios[1]:.2f}")
    logging.info(f"Test: {len(test_split)} images, Person ratio: {test_ratios[0]:.2f}, Face ratio: {test_ratios[1]:.2f}")

    return train_split, val_split, test_split

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

    if yolo_target == "person+face":
        # Use stratified_split_person_face to preserve both person and face ratios.
        train_split, val_split, test_split = stratified_split_person_face(total_images, annotation_folder, train_ratio, val_ratio)
        splits = {
            "train": train_split,
            "val": val_split,
            "test": test_split
        }
        logging.info(f"Person+Face split: Train {len(splits['train'])} images, Val {len(splits['val'])} images, Test {len(splits['test'])} images.")
    else:
        # Existing logic for other yolo_targets
        train_with, val_with, test_with = stratified_split(with_class, train_ratio, val_ratio)
        train_without, val_without, test_without = stratified_split(without_class, train_ratio, val_ratio)
        splits = {
        "train": train_with + train_without,
        "val": val_with + val_without,
        "test": test_with + test_without
        }
        logging.info(f"Train split: {len(splits['train'])} images ({len(train_with)} class, {len(train_without)} no-class).")
        logging.info(f"Val split: {len(splits['val'])} images ({len(val_with)} class, {len(val_without)} no-class).")
        logging.info(f"Test split: {len(splits['test'])} images ({len(test_with)} class, {len(test_without)} no-class).")

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
    logging.info(f"Starting dataset preparation for {model_target} {yolo_target}...")
    os.environ['OMP_NUM_THREADS'] = '10'
    
    if model_target == "yolo":
        label_path = YoloPaths.person_labels_input_dir if yolo_target == "person" else YoloPaths.face_labels_input_dir if yolo_target == "face" else YoloPaths.person_face_labels_input_dir if yolo_target == "person+face" else YoloPaths.gaze_labels_input_dir
        data_path = YoloPaths.person_data_input_dir if yolo_target == "person" else YoloPaths.face_data_input_dir if yolo_target == "face" else YoloPaths.person_face_data_input_dir if yolo_target == "person+face" else YoloPaths.gaze_data_input_dir        
        
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
    parser.add_argument("--yolo_target", type=str, choices=["person", "face", "gaze", "person+face"], help="YOLO target type (person or face).")
    args = parser.parse_args()
    
    main(args.model_target, args.yolo_target)
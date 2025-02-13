import shutil
import random
import logging
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
    
def get_total_number_of_annotated_frames(label_path: Path, image_folder: Path = DetectionPaths.images_input_dir) -> list:
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
    for annotation_file in label_path.glob('*.txt'):
        parts = annotation_file.stem.split('_')
        video_name = "_".join(parts[:8])
        video_names.add(video_name)
    logging.info(f"Found {len(video_names)} unique video names")

    # Step 2: Count total images
    for video_name in video_names:
        video_path = image_folder / video_name
        if video_path.exists() and video_path.is_dir():
            total_images.extend(os.listdir(video_path))
    return total_images   

def get_class_distribution(label_path: Path):
    """
    This function reads the label files and groups images based on their class distribution.
    
    For each label file, all class_ids in the file are examined at once.
    Class 2 is ignored in the grouping so that:
      - Images with only 0s (or 0s and 2s) are considered as "only persons".
      - Images with only 1s (or 1s and 2s) are considered as "only faces".
      - Images with both 0 and 1 (with or without 2) are "multiple classes".
      - Otherwise, if the file is empty or contains no 0 or 1, it's "neither".
      
    Parameters:
    ----------
    label_path: Path
        Path to the directory containing label files.
    
    Returns:
    -------
    images_only_person: set
        Set of image names with only class 0.
    images_only_face: set
        Set of image names with only class 1.
    images_multiple: set
        Set of image names with both classes 0 and 1.
    images_neither: set
        Set of image names with no classes or only class 2.
    """
    images_only_person = set()
    images_only_face = set()
    images_multiple = set()
    images_neither = set()
    
    for label_file in label_path.glob('*.txt'):
        with label_file.open('r') as f:
            labels = f.readlines()
        
        # If the file is empty, classify as neither.
        if not labels:
            images_neither.add(label_file.stem)
            continue
        
        # Get all class_ids from the file.
        class_ids = {int(line.split()[0]) for line in labels if line.split()}
        # Ignore class 2
        reduced_ids = class_ids - {2}
        
        if reduced_ids == {0}:
            images_only_person.add(label_file.stem)
        elif reduced_ids == {1}:
            images_only_face.add(label_file.stem)
        elif reduced_ids == {0, 1}:
            images_multiple.add(label_file.stem)
        else:
            images_neither.add(label_file.stem)
    
    total_num_images = len(get_total_number_of_annotated_frames(label_path))
    only_person_ratio = len(images_only_person) / total_num_images
    only_face_ratio = len(images_only_face) / total_num_images
    multiple_ratio = len(images_multiple) / total_num_images
    neither_ratio = len(images_neither) / total_num_images
    logging.info(f"Total number of annotated frames: {total_num_images}")
    logging.info(f"Class distribution: {len(images_only_person)} only person {only_person_ratio:.2f}, {len(images_only_face)} only face {only_face_ratio:.2f}, {len(images_multiple)} multiple {multiple_ratio:.2f}, {len(images_neither)} neither {neither_ratio:.2f}")
    return images_only_person, images_only_face, images_multiple, images_neither

def stratified_split(image_sets: list, train_ratio: float = TrainingConfig.train_test_split_ratio):
    """
    This function splits the images into train, val, and test sets based on the class distribution.
    
    Parameters:
    ----------
    image_sets: list
        List of sets containing image names grouped by class distribution.
    train_ratio: float
        The ratio of images to be used for training, default is 0.8.
    
    Returns:
    -------
    train: list
        List of image names for training.
    val: list
        List of image names for validation.
    test: list  
        List of image names for testing.
    """
    val_ratio = (1 - train_ratio) / 2
    train, val, test = [], [], []
    
    logging.info(f"Splitting {len(image_list)} images from category {image_set}.")
    for image_set in image_sets:
        image_list = list(image_set)
        random.shuffle(image_list)
        total = len(image_list)
        
        train_split = int(total * train_ratio)
        val_split = int(total * val_ratio)
        
        train_split = int(total * train_ratio)
        val_split = int(total * val_ratio)
        
        logging.info(f"Added {len(image_list[:train_split])} to train, {len(image_list[train_split:train_split + val_split])} to val, {len(image_list[train_split + val_split:])} to test.")
        train.extend(image_list[:train_split])
        val.extend(image_list[train_split:train_split + val_split])
        test.extend(image_list[train_split + val_split:])
    
    return train, val, test

def move_images(image_names: list, split_type: str, label_path: Path):
    """
    This function moves the images to the specified split directory.
    
    Parameters:
    ----------
    image_names: list
        List of image names to be moved.
    split_type: str
        The split type (train, val, or test).
    label_path: Path
        Path to the directory containing
    """
    logging.info(f"Moving {len(image_names)} images to {split_type} split...")
    image_dst_dir = Path(YoloPaths.person_face_output_dir) / "images" / split_type
    label_dst_dir = Path(YoloPaths.person_face_output_dir) / "labels" / split_type
    
    # Create directories if they don't exist
    image_dst_dir.mkdir(parents=True, exist_ok=True)
    label_dst_dir.mkdir(parents=True, exist_ok=True)
    
    for image_name in image_names:
        image_src = Path(DetectionPaths.images_input_dir) / f"{image_name}.jpg"
        label_src = label_path / f"{image_name}.txt"
        image_dst = image_dst_dir / f"{image_name}.jpg"
        label_dst = label_dst_dir / f"{image_name}.txt"
        
        # check if label file exists and create it if not
        if not label_dst.exists():
            label_dst.touch()
        shutil.copy(image_src, image_dst)
        shutil.copy(label_src, label_dst)

def split_yolo_data(label_path: Path, yolo_target: str):
    """
    This function prepares the dataset for YOLO training by splitting the images into train, val, and test sets.
    
    Parameters:
    ----------
    label_path: Path
        Path to the directory containing label files.
    yolo_target: str
        The target object for YOLO detection.
    """
    if yolo_target == "person_face":
        image_sets = get_class_distribution(label_path)
        train, val, test = stratified_split(image_sets)
        
        move_images(train, "train", label_path)
        move_images(val, "val", label_path)
        move_images(test, "test", label_path)
    elif yolo_target == "gaze":
        # Implement stratified split for gaze if needed
        pass

def main(model_target, yolo_target):
    if model_target == "yolo":
        label_path = Path(YoloPaths.person_face_labels_input_dir) if yolo_target == "person_face" else Path(YoloPaths.gaze_labels_input_dir)
        split_yolo_data(label_path, yolo_target)
        logging.info("Dataset preparation for YOLO completed.")
    elif model_target == "other_model":
        pass
    else:
        logging.error("Unsupported model target specified!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for model training.")
    parser.add_argument("--model_target", choices=["yolo", "other_model"], required=True, help="Specify the model type")
    parser.add_argument("--yolo_target", choices=["person_face", "gaze"], required=True, help="Specify the YOLO target type")
    
    args = parser.parse_args()
    main(args.model_target, args.yolo_target)
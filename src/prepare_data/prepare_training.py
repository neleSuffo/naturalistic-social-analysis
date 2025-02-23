import shutil
import random
import logging
import argparse
import numpy as np
from pathlib import Path
from constants import DetectionPaths, YoloPaths
from config import TrainingConfig
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

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
        The target type for YOLO (e.g., "person_face" or "gaze").
    """
    # Define paths based on model and YOLO target
    paths = {
        ("yolo", "person_face"): YoloPaths.person_face_data_input_dir,
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
            total_images.extend([str(p.resolve()) for p in video_path.iterdir() if p.is_file()])
    return total_images   

def get_pfo_class_distribution(total_images: list, annotation_folder: Path, yolo_target: str) -> tuple:
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
    total_images: list
        List of image names.
    annotation_folder: Path
        Path to the directory containing label files.
    yolo_target: str
        The target type for YOLO (e.g., "person_face" or "person_face_object").
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
    object_counts = {
        3: "book",
        4: "animal",
        5: "toy", 
        6: "kitchenware",
        7: "screen",
        8: "food",
        9: "other_object"
    }
        
    images_only_person = set()
    images_only_face = set()
    images_multiple = set()
    images_neither = set()
    image_objects = {}  # Stores object presence per image
                    
    for image_file in total_images:
        image_file = Path(image_file)
        annotation_file = annotation_folder / image_file.with_suffix('.txt').name
        present_objects = set()
        
        if annotation_file.exists() and annotation_file.stat().st_size > 0:
            with open(annotation_file, 'r') as f:
                labels = f.readlines()    
        
            # Get all class_ids from the file.
            class_ids = {int(line.split()[0]) for line in labels if line.split()}
            
            # Ignore class 2
            reduced_ids = class_ids - {2}
            
            if reduced_ids == {0}: # Only person
                images_only_person.add(image_file.stem)
            elif reduced_ids == {1}: # Only face
                images_only_face.add(image_file.stem)
            elif reduced_ids == {0, 1}: # only person and face
                images_multiple.add(image_file.stem)
            else:
                images_neither.add(image_file.stem)
        else:
            images_neither.add(image_file.stem)
        
    # log number of each object type
    for object_type in object_counts.values():
        object_count = sum(1 for objects in image_objects.values() if object_type in objects)
        logging.info(f"Number of images with {object_type}: {object_count}")
    
    # Store object labels for this image
    total_num_images = len(total_images)
    only_person_ratio = len(images_only_person) / total_num_images
    only_face_ratio = len(images_only_face) / total_num_images
    multiple_ratio = len(images_multiple) / total_num_images
    neither_ratio = len(images_neither) / total_num_images
    logging.info(f"Total number of annotated frames: {total_num_images}")
    logging.info(f"Class distribution: {len(images_only_person)} only person {only_person_ratio:.2f}, {len(images_only_face)} only face {only_face_ratio:.2f}, {len(images_multiple)} multiple {multiple_ratio:.2f}, {len(images_neither)} neither {neither_ratio:.2f}")

    image_objects[image_file.stem] = list(present_objects)    

    return images_only_person, images_only_face, images_multiple, images_neither, image_objects


def get_pf_class_distribution(total_images: list, annotation_folder: Path) -> tuple:
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
    total_images: list
        List of image names.
    annotation_folder: Path
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
    image_objects: dict
        Dictionary containing object categories for each image.
    """
    object_counts = {
        3: ["book", 0],
        4: ["animal", 0],
        5: ["toy", 0], 
        6: ["kitchenware", 0],
        7: ["screen", 0],
        8: ["food", 0],
        9: ["other_object", 0]
    }
        
    # Create reverse mapping from class_id to object name
    id_to_name = {k: v[0] for k, v in object_counts.items()}
    
    images_only_person = set()
    images_only_face = set()
    images_multiple = set()
    images_neither = set()
    image_objects = {}
    
    for image_file in total_images:
        image_file = Path(image_file)
        annotation_file = annotation_folder / image_file.with_suffix('.txt').name
        image_objects[image_file.stem] = []

        if annotation_file.exists() and annotation_file.stat().st_size > 0:
            with open(annotation_file, 'r') as f:
                labels = f.readlines()    
        
            # Get all class_ids from the file.
            class_ids = {int(line.split()[0]) for line in labels if line.split()}
            
            # Build list of object categories for this image
            image_objects[image_file.stem] = [id_to_name[class_id] 
                                            for class_id in class_ids 
                                            if class_id in id_to_name]
           # Update object counts
            for class_id in class_ids:
                if class_id in object_counts:
                    object_counts[class_id][1] += 1
            # Ignore class 2
            reduced_ids = class_ids - {2}
            
            if reduced_ids == {0}:
                images_only_person.add(image_file.stem)
            elif reduced_ids == {1}:
                images_only_face.add(image_file.stem)
            elif reduced_ids == {0, 1}:
                images_multiple.add(image_file.stem)
            else:
                images_neither.add(image_file.stem)
        else:
            images_neither.add(image_file.stem)
        
    total_num_images = len(total_images)
    only_person_ratio = len(images_only_person) / total_num_images
    only_face_ratio = len(images_only_face) / total_num_images
    multiple_ratio = len(images_multiple) / total_num_images
    neither_ratio = len(images_neither) / total_num_images
    logging.info(f"Total number of annotated frames: {total_num_images}")
    logging.info(f"Class distribution: {len(images_only_person)} only person {only_person_ratio:.2f}, {len(images_only_face)} only face {only_face_ratio:.2f}, {len(images_multiple)} multiple {multiple_ratio:.2f}, {len(images_neither)} neither {neither_ratio:.2f}")
    # Log object counts
    return images_only_person, images_only_face, images_multiple, images_neither, image_objects

def get_gaze_class_distribution(total_images: list, annotation_folder: Path) -> tuple:
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
    total_images: list
        List of image names.
    annotation_folder: Path
        Path to the directory containing label files.
    
    Returns:
    -------
    images_no_gaze: set
        Set of image names with class 0.
    images_gaze: set
        Set of image names with class 1.    
    """
    images_no_gaze= set()
    images_gaze = set()
    
    for image_file in total_images:
        image_file = Path(image_file)
        annotation_file = annotation_folder / image_file.with_suffix('.txt').name
        if annotation_file.exists() and annotation_file.stat().st_size > 0:
            with open(annotation_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    class_id = line.strip().split()[0]
                    if class_id == "0":  # 0 = no gaze
                        images_no_gaze.add(f"{image_file.stem}_face_{i}.jpg")
                    elif class_id == "1":  # 1 = gaze
                        images_gaze.add(f"{image_file.stem}_face_{i}.jpg")
    
    total_gaze = len(images_gaze) + len(images_no_gaze)
    ratio = len(images_gaze) / total_gaze if total_gaze > 0 else 0
    logging.info(f"Gaze-to-No-Gaze Ratio: {ratio:.4f} ({len(images_gaze)}/{len(images_no_gaze)})")
    
    return images_gaze, images_no_gaze

def gaze_stratified_split(image_sets: list, train_ratio: float = TrainingConfig.train_test_split_ratio):
    """
    This function splits the images into train, val, and test sets based on the class distribution.
    
    Parameters:
    ----------
    image_sets: list
        List of sets containing image names grouped by class distribution.
    yolo_target: str
        The target type for YOLO (e.g., "person_face" or "gaze").
    train_ratio: float
        The ratio of images to be used for training, default is 0.8.
    
    Returns:
    -------
    dict: {
        "gaze": (list of train, list of val, list of test),
        "no_gaze": (list of train, list of val, list of test)
    }
    """
    val_ratio = (1 - train_ratio) / 2
    
    # For gaze target, split each set separately
    if len(image_sets) != 2:
        raise ValueError("For yolo_target 'gaze', image_sets must contain exactly two sets (no_gaze and gaze).")
    result = {}
    class_labels = ["gaze", "no_gaze"]
    
    # Convert tuple elements to lists and undersample 'no_gaze' to match the number of 'gaze' images
    gaze_images = list(image_sets[0])
    no_gaze_images = list(image_sets[1])
    total_gaze = len(gaze_images)
    if len(no_gaze_images) >= total_gaze:
        no_gaze_images_new = random.sample(no_gaze_images, total_gaze)
    else:
        logging.warning("Not enough 'no_gaze' images; using all available.")
        
    new_image_sets = [gaze_images, no_gaze_images_new]

    for idx, label in enumerate(class_labels):
        image_list = list(new_image_sets[idx])
        logging.info(f"Splitting {len(image_list)} images for class '{label}'.")
        random.shuffle(image_list)
        total = len(image_list)
        train_split = int(total * train_ratio)
        val_split = int(total * val_ratio)
        train = image_list[:train_split]
        val = image_list[train_split:train_split + val_split]
        test = image_list[train_split + val_split:]
        logging.info(f"Class '{label}': {len(train)} train, {len(val)} val, {len(test)} test images.")
        result[label] = (train, val, test)
    return result

def stratified_split_with_objects(image_sets, image_objects, train_ratio=TrainingConfig.train_test_split_ratio):
    """
    Perform a two-stage split:
    1. Split person/face presence groups normally.
    2. Within the "neither" group, balance object categories using multi-label stratification.
    """
    train, val, test = [], [], []
    val_ratio = (1 - train_ratio) / 2
    
    # First split the three sets (only person, only face, both person & face) normally
    for image_set in image_sets[:-1]:  # Exclude "neither" for now
        image_list = list(image_set)
        random.shuffle(image_list)
        total = len(image_list)
        train_split = int(total * train_ratio)
        val_split = int(total * val_ratio)
        
        train.extend(image_list[:train_split])
        val.extend(image_list[train_split:train_split + val_split])
        test.extend(image_list[train_split + val_split:])
    
    # Now handle "neither" set using Multi-Label Stratified Sampling
    object_categories = ["book", "animal", "toy", "kitchenware", "screen", "food", "other_object"]
    images_neither = list(image_sets[-1])  # The "neither" set (last in list)
    
    # Create binary matrix (multi-label representation)
    y = np.zeros((len(images_neither), len(object_categories)), dtype=int)
    for i, img in enumerate(images_neither):
        for j, category in enumerate(object_categories):
            if category in image_objects.get(img, []):
                y[i, j] = 1  # Mark object presence

    # First split (train vs rest)
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, temp_idx = next(msss.split(images_neither, y))

    # Second split (val vs test) - split the remaining 20% equally
    test_val_images = np.array(images_neither)[temp_idx]
    test_val_labels = y[temp_idx]
    msss_second = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(msss_second.split(test_val_images, test_val_labels))

    # Convert indices back to original space
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]
    
    # After splits are created, count and log object distributions
    def count_objects(image_list):
        counts = {cat: 0 for cat in object_categories}
        for img in image_list:
            for obj in image_objects.get(img, []):
                counts[obj] += 1
        return counts
    
    # Get neither-set images for each split
    neither_train = [images_neither[i] for i in train_idx]
    neither_val = [images_neither[i] for i in val_idx]
    neither_test = [images_neither[i] for i in test_idx]

    # Count objects in each split
    train_counts = count_objects(neither_train)
    val_counts = count_objects(neither_val)
    test_counts = count_objects(neither_test)

    # Log the distributions
    logging.info("Object distribution in 'neither' set splits:")
    logging.info(f"{'Category':<12} {'Train':<8} {'Val':<8} {'Test':<8}")
    logging.info("-" * 36)
    for category in object_categories:
        logging.info(f"{category:<12} {train_counts[category]:<8} {val_counts[category]:<8} {test_counts[category]:<8}")

    # Assign final splits
    train.extend([images_neither[i] for i in train_idx])
    val.extend([images_neither[i] for i in val_idx])
    test.extend([images_neither[i] for i in test_idx])

    return train, val, test

def log_split_distributions(train, val, test, image_objects, image_sets):
    """Log detailed distribution of images across splits"""
    
    # Log people/face distributions
    splits = {'Train': train, 'Val': val, 'Test': test}
    categories = {
        'Person only': image_sets[0],
        'Face only': image_sets[1], 
        'Person & Face': image_sets[2],
        'Neither (objects)': image_sets[3]
    }

    logging.info("\nImage Distribution by Category:")
    logging.info(f"{'Category':<15} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    logging.info("-" * 47)
    
    for cat_name, cat_set in categories.items():
        train_count = len(set(train) & cat_set)
        val_count = len(set(val) & cat_set) 
        test_count = len(set(test) & cat_set)
        total = len(cat_set)
        logging.info(f"{cat_name:<15} {train_count:<8} {val_count:<8} {test_count:<8} {total:<8}")

    # Log object distributions
    object_categories = ["book", "animal", "toy", "kitchenware", "screen", "food", "other_object"]
    
    logging.info("\nObject Distribution in 'Neither' Category:")
    logging.info(f"{'Object':<12} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    logging.info("-" * 44)
    
    for obj in object_categories:
        train_count = sum(1 for img in train if obj in image_objects.get(img, []))
        val_count = sum(1 for img in val if obj in image_objects.get(img, []))
        test_count = sum(1 for img in test if obj in image_objects.get(img, []))
        total = train_count + val_count + test_count
        logging.info(f"{obj:<12} {train_count:<8} {val_count:<8} {test_count:<8} {total:<8}")
        
def move_images(yolo_target: str, image_names: list, split_type: str, label_path: Path):
    """
    This function moves the images to the specified split directory.
    
    Parameters:
    ----------
    yolo_target: str
        The target type for YOLO (e.g., "person_face","person_face_object", "gaze" or "no_gaze").
    image_names: list
        List of image names to be moved.
    split_type: str
        The split type (train, val, or test).
    label_path: Path
        Path to the directory containing
    """
    logging.info(f"Moving {len(image_names)} images to {split_type} split...")
    if yolo_target == "person_face":
        image_dst_dir = YoloPaths.person_face_data_input_dir / "images" / split_type
        label_dst_dir = YoloPaths.person_face_data_input_dir / "labels" / split_type
    elif yolo_target == "gaze" or yolo_target == "no_gaze":
        image_dst_dir = YoloPaths.gaze_data_input_dir / split_type / yolo_target
        label_dst_dir = YoloPaths.gaze_data_input_dir /  split_type / yolo_target
    elif yolo_target == "person_face_object":
        image_dst_dir = YoloPaths.person_face_object_data_input_dir / "images" / split_type
        label_dst_dir = YoloPaths.person_face_object_data_input_dir / "labels" / split_type
    
    # Create directories if they don't exist
    image_dst_dir.mkdir(parents=True, exist_ok=True)
    label_dst_dir.mkdir(parents=True, exist_ok=True)
    
    for image_name in image_names:
        if yolo_target == "person_face" or yolo_target == "person_face_object":
            # construct full image path
            image_parts = image_name.split("_")[:8]
            image_folder = "_".join(image_parts)
            image_src = DetectionPaths.images_input_dir / image_folder / f"{image_name}.jpg"
            label_src = label_path / f"{image_name}.txt"
            image_dst = image_dst_dir / f"{image_name}.jpg"
            label_dst = label_dst_dir / f"{image_name}.txt"
            # check if label file exists and create it if not
            if not label_src.exists():
                label_dst.touch()  # Create an empty destination label file if the source does not exist
            else:
                shutil.copy(label_src, label_dst)
            shutil.copy(image_src, image_dst)
        
        elif yolo_target == "gaze" or yolo_target == "no_gaze":
            image_src = DetectionPaths.gaze_images_input_dir  / image_name
            image_dst = image_dst_dir / image_name

            if not image_src.exists():
                logging.warning(f"Image {image_src} does not exist. Skipping...")
                continue
            shutil.copy(image_src, image_dst)

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
    total_images = get_total_number_of_annotated_frames(label_path)
    
     # Mapping of target type to corresponding distribution function.
    distribution_funcs = {
        "person_face": get_pf_class_distribution,
        "person_face_object": get_pf_class_distribution,
        "gaze": get_gaze_class_distribution,
    }
    try:
        if yolo_target == "gaze":
            images_gaze, images_no_gaze = distribution_funcs[yolo_target](total_images, label_path)
        else:
            images_only_person, images_only_face, images_multiple, images_neither, image_objects = distribution_funcs[yolo_target](total_images, label_path)
    except KeyError:
        raise ValueError(f"Invalid yolo_target: {yolo_target}")
    
    if yolo_target == "gaze":
        splits_dict = gaze_stratified_split((looking_at, not_looking_at))
        for gaze_class in ["gaze", "no_gaze"]:
            train, val, test = splits_dict[gaze_class]
            for split_name, split_set in (("train", train), ("val", val), ("test", test)):
                move_images(gaze_class, split_set, split_name, label_path)
    elif yolo_target == "person_face":
        train, val, test = stratified_split((images_only_person, images_only_face, images_multiple, images_neither))
        for split_name, split_set in (("train", train), ("val", val), ("test", test)):
            move_images(yolo_target, split_set, split_name, label_path)
    elif yolo_target == "person_face_object":
        train, val, test = stratified_split_with_objects(
            (images_only_person, images_only_face, images_multiple, images_neither), 
            image_objects
        )        
        image_sets = (images_only_person, images_only_face, images_multiple, images_neither)
        log_split_distributions(train, val, test, image_objects, image_sets)
        for split_name, split_set in (("train", train), ("val", val), ("test", test)):
            move_images(yolo_target, split_set, split_name, label_path)
        
def main(model_target: str, yolo_target: str):
    """
    Main function to prepare the dataset for model training.
    
    Parameters:
    ----------
    model_target : str
        The target model for preparation (e.g., "yolo").
    yolo_target : str
        The target type for YOLO (e.g., "person" or "face").
    """
    if model_target == "yolo":
        label_path = YoloPaths.person_face_labels_input_dir if yolo_target == "person_face" else YoloPaths.person_face_object_labels_input_dir if yolo_target == "person_face_object" else YoloPaths.gaze_labels_input_dir
        split_yolo_data(label_path, yolo_target)
        logging.info("Dataset preparation for YOLO completed.")
    elif model_target == "other_model":
        pass
    else:
        logging.error("Unsupported model target specified!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for model training.")
    parser.add_argument("--model_target", choices=["yolo", "other_model"], required=True, help="Specify the model type")
    parser.add_argument("--yolo_target", choices=["person_face", "person_face_object", "gaze"], required=True, help="Specify the YOLO target type")
    
    args = parser.parse_args()
    main(args.model_target, args.yolo_target)
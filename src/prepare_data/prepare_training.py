import shutil
import random
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
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
     
def get_total_number_of_annotated_frames(label_path: Path, image_folder: Path = DetectionPaths.images_input_dir, target_type: str = None) -> list:
    """
    This function returns the total number of annotated frames in the dataset.
    
    Parameters
    ----------
    label_path : Path
        the path to the label files
    image_folder : Path
        the path to the image folder
    target_type : str
        the target type for YOLO (e.g., "child_person_face" or "adult_person_face")
        
    Returns
    -------
    list
        the total number of images
    """
    positive_frames = set()  # Frames with target annotations
    negative_frames = set()  # Frames without target annotations
    video_names = set()
    total_images = []
    
    # Step 1: Get unique video names
    for annotation_file in label_path.glob('*.txt'):
        parts = annotation_file.stem.split('_')
        video_name = "_".join(parts[:8])
        video_names.add(video_name)

        # Read the annotation file
        with open(annotation_file, 'r') as f:
            annotations = f.readlines()
            
        has_target = False
        if target_type in ["child_person_face", "adult_person_face"]:
            # Check if file contains adult/child annotations (class 0 or 1)
            has_target = any(line.startswith(('0', '1')) for line in annotations)
        elif target_type == 'child_person_face':
            # Check if file contains child annotations (class 2 or 3)
            has_target = any(line.startswith(('0', '1', '2')) for line in annotations)
            
        if has_target:
            positive_frames.add(annotation_file.stem)
        else:
            negative_frames.add(annotation_file.stem)
    
    logging.info(f"Found {len(video_names)} unique video names")

    
    # Step 2: Count total images
    for video_name in video_names:
        video_path = image_folder / video_name
        if video_path.exists() and video_path.is_dir():
            total_images.extend([str(p.resolve()) for p in video_path.iterdir() if p.is_file()])
    return total_images   

def get_pf_class_distribution(total_images: list, annotation_folder: Path, target_type: str) -> pd.DataFrame:
    """
    Reads label files and groups images based on their person/face class distribution.

    Parameters:
    ----------
    total_images: list
        List of image names.
    annotation_folder: Path
        Path to the directory containing label files.
    target_type: str
        The target type for YOLO (e.g., "person_face" or "adult_person_face").
        
    Returns:
    -------
    df: pd.DataFrame
        DataFrame containing image filenames and their corresponding one-hot encoded class labels.
    """
    # Define the classes based on target type
    if target_type == "adult_person_face":
        class_mapping = {
            0: "adult_person",
            1: "adult_face",
        }
    elif target_type == "child_person_face":
        class_mapping = {
            0: "child_person",
            1: "child_face",
            2: "child_body_parts"
        }
    else:
        class_mapping = {
            0: "person",
            1: "face",
        }

    image_class_mapping = []

    for image_file in total_images:
        image_file = Path(image_file)
        annotation_file = annotation_folder / image_file.with_suffix('.txt').name

        if annotation_file.exists() and annotation_file.stat().st_size > 0:
            with open(annotation_file, 'r') as f:
                class_ids = {int(line.split()[0]) for line in f if line.split()}

            # Convert class IDs to names
            labels = [class_mapping[cid] for cid in class_ids if cid in class_mapping]
        else:
            labels = []

        # Create one-hot encoded dictionary
        image_class_mapping.append({
            "filename": image_file.stem,
            **{class_name: (1 if class_name in labels else 0) for class_name in class_mapping.values()}
        })

    # Convert to DataFrame
    df = pd.DataFrame(image_class_mapping)
    return df

def get_all_class_distribution(total_images: list, annotation_folder: Path):
    """
    Reads label files and groups images based on their refined class distribution.

    Parameters:
    ----------
    total_images: list
        List of image names.
    annotation_folder: Path
        Path to the directory containing label files.

    Returns:
    -------
    df: pd.DataFrame
        DataFrame containing image filenames and their corresponding one-hot encoded class labels.
    """
    object_mapping = {
        5: "book",
        6: "toy", 
        7: "kitchenware",
        8: "screen",
        9: "food",
        10: "other_object",
    }

    # Class mapping
    person_mapping = {
        0: "adult_person",
        1: "child_person",
        2: "adult_face",
        3: "child_face"
    }

    # Combine all class mappings
    id_to_name = {**person_mapping, **object_mapping}

    image_class_mapping = []

    for image_file in total_images:
        image_file = Path(image_file)
        annotation_file = annotation_folder / image_file.with_suffix('.txt').name

        if annotation_file.exists() and annotation_file.stat().st_size > 0:
            with open(annotation_file, 'r') as f:
                class_ids = {int(line.split()[0]) for line in f if line.split()}

            # Convert class IDs to names
            labels = [id_to_name[cid] for cid in class_ids if cid in id_to_name]
        else:
            labels = []

        image_class_mapping.append({
            "filename": image_file.stem,
            **{class_name: (1 if class_name in labels else 0) for class_name in id_to_name.values()}
        })

    # Convert to DataFrame
    df = pd.DataFrame(image_class_mapping)

    # save df
    return df

def get_object_class_distribution(total_images: list, annotation_folder: Path, target_type: bool = None) -> pd.DataFrame:
    """
    Reads label files and groups images based on their refined class distribution.

    Parameters:
    ----------
    total_images: list
        List of image names.
    annotation_folder: Path
        Path to the directory containing label files.

    Returns:
    -------
    df: pd.DataFrame
        DataFrame containing image filenames and their corresponding one-hot encoded class labels.
    """
    object_mapping = {
        0: "book",
        1: "toy", 
        2: "kitchenware",
        3: "screen",
        4: "other_object",
        5: "animal",
        6: "food"
    }

    image_class_mapping = []

    for image_file in total_images:
        image_file = Path(image_file)
        annotation_file = annotation_folder / image_file.with_suffix('.txt').name

        if annotation_file.exists() and annotation_file.stat().st_size > 0:
            with open(annotation_file, 'r') as f:
                class_ids = {int(line.split()[0]) for line in f if line.split()}

            # Convert class IDs to names
            labels = [object_mapping[cid] for cid in class_ids if cid in object_mapping]
        else:
            labels = []

        image_class_mapping.append({
            "filename": image_file.stem,
            **{class_name: (1 if class_name in labels else 0) for class_name in object_mapping.values()}
        })

    # Convert to DataFrame
    df = pd.DataFrame(image_class_mapping)

    # save df
    return df

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
    
def stratified_split_all(df: pd.DataFrame, train_ratio=TrainingConfig.train_test_split_ratio, random_seed=TrainingConfig.random_seed, yolo_target=None):
    """
    Splits the dataset into training, validation, and testing while preserving class distribution.

    Parameters:
    ----------
    df: pd.DataFrame
        DataFrame containing image filenames and their one-hot encoded class labels.
    train_ratio: float
        Ratio of the dataset to allocate for training (default from TrainingConfig).
    random_seed: int
        Random seed for reproducibility.
    yolo_target: str
        The target type for YOLO (e.g., "adult_person_face", "object", "all").

    Returns:
    -------
    train_df, val_df, test_df: pd.DataFrame
        DataFrames containing train, validation, and test splits.
    """
    val_test_ratio = (1 - train_ratio) / 2  # Split remaining data equally between val & test

    # Extract multi-labels (excluding first column if it's filenames)
    y = df.iloc[:, 1:].values  

    # First split (train vs rest)
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=(1 - train_ratio), random_state=random_seed)
    train_idx, temp_idx = next(msss.split(df, y))

    # Get initial training data
    train_df = df.iloc[train_idx].copy()
    
    # Balance training set only for person face detection
    if yolo_target in ['adult_person_face', 'child_person_face', "object"]:
        # Sum relevant columns based on target type
        if yolo_target == 'adult_person_face':
            train_df['has_annotation'] = (train_df['adult_person'] + train_df['adult_face']) > 0
        elif yolo_target == "child_person_face":
            train_df['has_annotation'] = (train_df['child_person'] + train_df['child_face'] + train_df['child_body_parts']) > 0
        else:  # object
            train_df['has_annotation'] = train_df['book'] + train_df['toy'] + train_df['kitchenware'] + train_df['screen'] + train_df['food'] + train_df['other_object'] + train_df['animal'] > 0
        
        positive_samples = train_df[train_df['has_annotation']].copy()
        negative_samples = train_df[~train_df['has_annotation']].copy()
        
        n_positive = len(positive_samples)
        logging.info(f"Initial training set distribution for {yolo_target}:")
        logging.info(f"Positive samples: {n_positive}")
        logging.info(f"Negative samples: {len(negative_samples)}")
        
        # Subsample negative samples to match positive samples
        if len(negative_samples) > n_positive:
            negative_samples = negative_samples.sample(n=n_positive, random_state=random_seed)
            logging.info(f"Downsampled negative samples to {n_positive}")
        
        # Combine balanced samples
        train_df = pd.concat([positive_samples, negative_samples])
        train_df = train_df.drop('has_annotation', axis=1)
        logging.info(f"Final balanced training set size: {len(train_df)}")
        
    # Second split (val vs test) - split remaining 20% equally
    msss_second = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=random_seed)
    val_idx, test_idx = next(msss_second.split(df.iloc[temp_idx], y[temp_idx]))

    # Get final DataFrames
    val_df = df.iloc[temp_idx[val_idx]]
    test_df = df.iloc[temp_idx[test_idx]]
    
    # get file names from the first column as list
    train = train_df.iloc[:, 0].tolist()
    val = val_df.iloc[:, 0].tolist()
    test = test_df.iloc[:, 0].tolist()
    
    return train, val, test, train_df, val_df, test_df

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
    object_categories = ["book", "toy", "kitchenware", "screen", "food", "other_object", "animal"]
    
    logging.info("\nObject Distribution in 'Neither' Category:")
    logging.info(f"{'Object':<12} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    logging.info("-" * 44)
    
    for obj in object_categories:
        train_count = sum(1 for img in train if obj in image_objects.get(img, []))
        val_count = sum(1 for img in val if obj in image_objects.get(img, []))
        test_count = sum(1 for img in test if obj in image_objects.get(img, []))
        total = train_count + val_count + test_count
        logging.info(f"{obj:<12} {train_count:<8} {val_count:<8} {test_count:<8} {total:<8}")
 
def log_all_split_distributions(train_df, val_df, test_df, yolo_target):
    """Log detailed distribution of images across splits.
    
    Parameters:
    ----------
    train_df: pd.DataFrame
        DataFrame containing training split.
    val_df: pd.DataFrame
        DataFrame containing validation split.
    test_df: pd.DataFrame
        DataFrame containing testing split.
    yolo_target: str
        The target type for YOLO (e.g., "adult_person_face", "child_person_face" or "all").
    """

    splits = {'Train': train_df, 'Val': val_df, 'Test': test_df}

    # All class categories
    if yolo_target == "adult_person_face":
        all_categories = ["adult_person", "adult_face"]
    elif yolo_target == "child_person_face":
        all_categories = ["child_person", "child_face"]
    elif yolo_target == "object":
        all_categories = ["book", "toy", "kitchenware", "screen", "other_object", "animal", "food"]
    else:
        all_categories = [
            "adult_person", "child_person", "adult_face", "child_face",
            "book", "toy", "kitchenware", "screen", "food", "other_object"
        ]

    logging.info("\nImage Distribution by Category:")
    logging.info(f"{'Category':<20} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    logging.info("-" * 55)

    for cat_name in all_categories:
        train_count = train_df[cat_name].sum()
        val_count = val_df[cat_name].sum()
        test_count = test_df[cat_name].sum()
        total = train_count + val_count + test_count
        logging.info(f"{cat_name:<20} {train_count:<8} {val_count:<8} {test_count:<8} {total:<8}")
    
    # log number of images per split
    logging.info(f"Total number of images: {len(train_df) + len(val_df) + len(test_df)}")
    logging.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

def stratified_split(image_sets, train_ratio=TrainingConfig.train_test_split_ratio):
    """Basic stratified split without object consideration"""
    train, val, test = [], [], []
    val_ratio = (1 - train_ratio) / 2
    
    for image_set in image_sets:
        image_list = list(image_set)
        random.shuffle(image_list)
        total = len(image_list)
        train_split = int(total * train_ratio)
        val_split = int(total * val_ratio)
        
        train.extend(image_list[:train_split])
        val.extend(image_list[train_split:train_split + val_split])
        test.extend(image_list[train_split + val_split:])
    
    return train, val, test
          
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
        
    Raises
    ------
    ValueError
        If yolo_target is invalid or paths cannot be determined
    """
    if not image_names:
        logging.info(f"No images to move for {yolo_target} {split_type}")
        return
    
    # Get destination paths from configuration
    paths = YoloPaths.get_target_paths(yolo_target, split_type)
    if not paths:
        raise ValueError(f"Invalid yolo_target: {yolo_target}")
    
    image_dst_dir, label_dst_dir = paths

    # Create directories if they don't exist
    image_dst_dir.mkdir(parents=True, exist_ok=True)
    label_dst_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Moving {len(image_names)} images to {split_type} split...")
    
    for image_name in image_names:
        try:
            if yolo_target in ["all", "child_person_face", "adult_person_face", "object"]:
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
                    shutil.copy2(label_src, label_dst)
                    # Copy image if it exists
                if image_src.exists():
                    shutil.copy2(image_src, image_dst)
                else:
                    logging.warning(f"Image {image_src} does not exist. Skipping...")
                    continue    
                    
            elif yolo_target == "gaze" or yolo_target == "no_gaze":
                image_src = DetectionPaths.gaze_images_input_dir  / image_name
                image_dst = image_dst_dir / image_name

                if not image_src.exists():
                    logging.warning(f"Image {image_src} does not exist. Skipping...")
                    continue
                
                shutil.copy2(image_src, image_dst)
                
        except Exception as e:
            logging.error(f"Error processing {image_name}: {str(e)}")
            continue
        
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
        "all": get_all_class_distribution,
        "gaze": get_gaze_class_distribution,
        "adult_person_face": get_pf_class_distribution,
        "child_person_face": get_pf_class_distribution,
        "object": get_object_class_distribution,
    }
    try:
        if yolo_target == "gaze":
            images_gaze, images_no_gaze = distribution_funcs[yolo_target](total_images, label_path)
            splits_dict = gaze_stratified_split((images_gaze, images_no_gaze))
            for gaze_class in ["gaze", "no_gaze"]:
                train, val, test = splits_dict[gaze_class]
                for split_name, split_set in (("train", train), ("val", val), ("test", test)):
                    move_images(gaze_class, split_set, split_name, label_path)
        elif yolo_target in ["all", "adult_person_face", "child_person_face", "object"]:
            df = distribution_funcs[yolo_target](total_images, label_path, yolo_target)
            train, val, test, train_df, val_df, test_df = stratified_split_all(df, yolo_target=yolo_target)
            log_all_split_distributions(train_df, val_df, test_df, yolo_target)
            for split_name, split_set in (("train", train), ("val", val), ("test", test)):
                move_images(yolo_target, split_set, split_name, label_path)
    except KeyError:
        logging.error(f"Distribution function not found for target: {yolo_target}")
        raise ValueError(f"Invalid yolo_target: {yolo_target}")
            
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
        path_mapping = {
            "all": YoloPaths.all_labels_input_dir,
            "child_person_face": YoloPaths.child_person_face_labels_input_dir,
            "adult_person_face": YoloPaths.adult_person_face_labels_input_dir,
            "object": YoloPaths.object_labels_input_dir,
            "gaze": YoloPaths.gaze_labels_input_dir
        }
        label_path = path_mapping[yolo_target]
        split_yolo_data(label_path, yolo_target)
        logging.info("Dataset preparation for YOLO completed.")
    elif model_target == "other_model":
        pass
    else:
        logging.error("Unsupported model target specified!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for model training.")
    parser.add_argument("--model_target", choices=["yolo", "other_model"], required=True, help="Specify the model type")
    parser.add_argument("--yolo_target", choices=["all", "gaze", "adult_person_face", "child_person_face", "object"], required=True, help="Specify the YOLO target type")
    
    args = parser.parse_args()
    main(args.model_target, args.yolo_target)
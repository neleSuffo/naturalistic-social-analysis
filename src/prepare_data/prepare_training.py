import shutil
import random
import logging
import argparse
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from sklearn.model_selection import train_test_split
from constants import DetectionPaths, YoloPaths, ResNetPaths, BasePaths
from config import TrainingConfig
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit as MSSS
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Optional, Tuple
from sklearn.model_selection import StratifiedShuffleSplit

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_total_number_of_annotated_frames(label_path: Path, image_folder: Path = DetectionPaths.images_input_dir, target_type: str = None) -> list:
    """
    This function returns the total number of annotated frames in the dataset.
    
    Parameters
    ----------
    label_path : Path
        the path to the label files
    image_folder : Path
        the path to the image folder
    target_type : str, optional
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

def get_class_distribution(total_images: list, annotation_folder: Path, target_type: str = "all") -> pd.DataFrame:
    """
    Reads label files and groups images based on their class distribution.

    Parameters:
    ----------
    total_images: list
        List of image names.
    annotation_folder: Path
        Path to the directory containing label files.
    target_type: str
        The target type for classification (e.g., "all", "person_face", "object", etc.)

    Returns:
    -------
    pd.DataFrame
        DataFrame containing image filenames and their corresponding one-hot encoded class labels.
    """
    # Define class mappings based on target type
    class_mappings = {
        "all": {
            0: "adult_person", 1: "child_person", 2: "adult_face", 3: "child_face",
            5: "book", 6: "toy", 7: "kitchenware", 8: "screen", 
            9: "other_object", 10: "other_object" # map former food and animal class to other_object
        },
        "person_face": {
            0: "person", 1: "face", 2: "child_body_parts"
        },
        "adult_person_face": {
            0: "adult_person", 1: "adult_face"
        },
        "child_person_face": {
            0: "child_person", 1: "child_face", 2: "child_body_parts"
        },
        "object": {
            0: "interacted_book", 1: "interacted_toy", 2: "interacted_kitchenware",
            3: "interacted_screen", 4: "interacted_other_object", 5: "book",
            6: "toy", 7: "kitchenware", 8: "screen", 9: "other_object",
            10: "interacted_animal", 11: "interacted_food", 12: "other_object", 13: "other_object" # map former food and animal class to other_object
        },
        "person_face_object": {
            0: "person", 1: "face", 2: "child_body_parts", 3: "book",
            4: "toy", 5: "kitchenware", 6: "screen", 7: "other_object",
            8: "other_object", 9: "other_object" # map former food and animal class to other_object
        }
    }

    # Get the appropriate class mapping
    if target_type not in class_mappings:
        raise ValueError(f"Invalid target_type: {target_type}")
    
    id_to_name = class_mappings[target_type]
    image_class_mapping = []

    for image_file in total_images:
        image_file = Path(image_file)
        annotation_file = annotation_folder / image_file.with_suffix('.txt').name

        # Get labels from annotation file
        labels = []
        if annotation_file.exists() and annotation_file.stat().st_size > 0:
            with open(annotation_file, 'r') as f:
                class_ids = {int(line.split()[0]) for line in f if line.split()}
                labels = [id_to_name[cid] for cid in class_ids if cid in id_to_name]

        # Create one-hot encoded dictionary
        image_class_mapping.append({
            "filename": image_file.stem,
            **{class_name: (1 if class_name in labels else 0) 
               for class_name in id_to_name.values()}
        })

    return pd.DataFrame(image_class_mapping)

def get_binary_class_distribution(total_images: list, annotation_folder: Path, target: str) -> Tuple[set, set]:
    images_class_0 = set()
    images_class_1 = set()
    
    class_description = "face" if target in ["gaze", "face"] else "person"
    for image_file in total_images:
        image_file = Path(image_file)
        annotation_file = annotation_folder / image_file.with_suffix('.txt').name
        if annotation_file.exists() and annotation_file.stat().st_size > 0:
            with open(annotation_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    class_id = line.strip().split()[0]
                    # Include suffix to distinguish detections
                    img_name = f"{image_file.stem}_{class_description}_{i}"
                    if class_id == "0":
                        images_class_0.add(img_name)
                    elif class_id == "1":
                        images_class_1.add(img_name)
    
    total = len(images_class_0) + len(images_class_1)
    logging.info(f"Class 0: {len(images_class_0)} images")
    logging.info(f"Class 1: {len(images_class_1)} images")
    logging.info(f"Total: {total} images")
    
    return images_class_0, images_class_1

def binary_stratified_split(image_sets: Tuple[set, set], yolo_target: str, 
                            test_size: float = 0.2, val_size: float = 0.2, 
                            random_state: int = 42) -> dict:
    """
    Perform a stratified split on binary class data while ensuring:
    - Test and validation sets maintain the original ratio of class 1 and class 0.
    - Training set is balanced 50/50.
    - All images from the same ID stay in one split.

    Parameters
    ----------
    image_sets : Tuple[set, set]
        Tuple of (images_class_0, images_class_1), sets of image names.
    yolo_target : str
        YOLO target type (e.g., "person", "face", "gaze").
    test_size : float
        Fraction of data for the test set.
    val_size : float
        Fraction of data for the validation set (from remaining data).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        {0: (train_list, val_list, test_list), 1: (train_list, val_list, test_list)}
    """
    images_class_0, images_class_1 = image_sets
    
    # Extract ID from image name
    def extract_id(image_name: str) -> str:
        parts = image_name.split('_')
        for part in parts:
            if part.startswith('id'):
                return part[2:]
        raise ValueError(f"Could not extract ID from {image_name}")

    # Create DataFrame with image names, classes, and IDs
    all_images = [(img, 0) for img in images_class_0] + [(img, 1) for img in images_class_1]
    df = pd.DataFrame(all_images, columns=['filename', 'class'])
    df['id'] = df['filename'].apply(extract_id)
    
    # Group by ID to ensure all images from an ID stay together
    id_df = df.groupby('id').agg({
        'filename': list,
        'class': list
    }).reset_index()
    id_df['class_label'] = id_df['class'].apply(lambda x: 0 if all(c == 0 for c in x) else 1 if all(c == 1 for c in x) else 2)  # 2 for mixed

    # Step 1: Split into train+val and test
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(sss_test.split(id_df.index, id_df['class_label']))

    # Step 2: Split train+val into train and val
    train_val_df = id_df.iloc[train_val_idx]
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=random_state)
    train_idx, val_idx = next(sss_val.split(train_val_df.index, train_val_df['class_label']))

    # Extract IDs for each split
    train_ids = train_val_df.iloc[train_idx]['id'].tolist()
    val_ids = train_val_df.iloc[val_idx]['id'].tolist()
    test_ids = id_df.iloc[test_idx]['id'].tolist()

    # Map IDs back to images
    id_to_images = dict(zip(id_df['id'], id_df['filename']))
    train = [img for id_ in train_ids for img in id_to_images[id_]]
    val = [img for id_ in val_ids for img in id_to_images[id_]]
    test = [img for id_ in test_ids for img in id_to_images[id_]]

    # Balance training set to 50/50
    train_class_0 = [img for img in train if img in images_class_0]
    train_class_1 = [img for img in train if img in images_class_1]
    min_class_size = min(len(train_class_0), len(train_class_1))
    if min_class_size > 0:
        train_class_0_balanced = random.sample(train_class_0, min_class_size, random_state=random_state)
        train_class_1_balanced = random.sample(train_class_1, min_class_size, random_state=random_state)
    else:
        train_class_0_balanced = train_class_0
        train_class_1_balanced = train_class_1
    train_balanced = train_class_0_balanced + train_class_1_balanced

    # Prepare result
    result = {
        0: ([img for img in train_balanced if img in images_class_0], 
            [img for img in val if img in images_class_0], 
            [img for img in test if img in images_class_0]),
        1: ([img for img in train_balanced if img in images_class_1], 
            [img for img in val if img in images_class_1], 
            [img for img in test if img in images_class_1])
    }

    # Logging
    datetime_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    BasePaths.logging_dir.mkdir(exist_ok=True)
    log_file = BasePaths.logging_dir / f"split_distribution_{yolo_target}_{datetime_str}.txt"
    
    train_ids_set = set(train_ids)
    val_ids_set = set(val_ids)
    test_ids_set = set(test_ids)
    overlap = train_ids_set & val_ids_set | train_ids_set & test_ids_set | val_ids_set & test_ids_set
    
    distribution_info = [
        f"Dataset split distribution for {yolo_target}",
        f"Original Class 0: {len(images_class_0)}, Class 1: {len(images_class_1)}",
        f"Train Class 0: {len(result[0][0])}, Class 1: {len(result[1][0])} (balanced 50/50)",
        f"Val Class 0: {len(result[0][1])}, Class 1: {len(result[1][1])}",
        f"Test Class 0: {len(result[0][2])}, Class 1: {len(result[1][2])}",
        f"Unique IDs - Train: {len(train_ids_set)}, Val: {len(val_ids_set)}, Test: {len(test_ids_set)}",
        f"ID overlap: {len(overlap)} (should be 0)"
    ]
    
    with open(log_file, 'w') as f:
        f.write('\n'.join(distribution_info))
    logging.info(f"Split distribution saved to: {log_file}")
    
    if overlap:
        logging.warning(f"ID overlap detected: {overlap}")
    
    return result

def balance_training_set(df: pd.DataFrame, yolo_target: str) -> pd.DataFrame:
    """Helper function to balance training set based on annotations."""
    # Define target columns based on yolo_target
    target_columns = get_target_columns(yolo_target)
    
    # Create has_annotation column
    df['has_annotation'] = df[target_columns].sum(axis=1) > 0
    
    pos_samples = df[df['has_annotation']].copy()
    neg_samples = df[~df['has_annotation']].copy()
    
    # Balance negative samples to match positive
    if len(neg_samples) > len(pos_samples):
        neg_samples = neg_samples.sample(
            n=len(pos_samples),
            random_state=TrainingConfig.random_seed
        )
    
    return pd.concat([pos_samples, neg_samples]).drop('has_annotation', axis=1)

def verify_split_distributions(train_df: pd.DataFrame,
                             val_df: pd.DataFrame,
                             test_df: pd.DataFrame):
    """Verify and log split distributions."""
    total = len(train_df) + len(val_df) + len(test_df)
    logging.info("\nSplit sizes:")
    logging.info(f"Train: {len(train_df)} ({len(train_df)/total:.2%})")
    logging.info(f"Val: {len(val_df)} ({len(val_df)/total:.2%})")
    logging.info(f"Test: {len(test_df)} ({len(test_df)/total:.2%})")
     
def multilabel_stratified_split(df: pd.DataFrame,
                              train_ratio: float = 0.8,
                              random_seed: int = TrainingConfig.random_seed,
                              yolo_target: str = None):
    """
    Performs stratified split maintaining 80/10/10 ratio.
    Only balances training set to preserve real-world distribution in val/test.
    """
    if df.empty:
        raise ValueError("Empty DataFrame provided")

    if 'filename' not in df.columns:
        raise ValueError("DataFrame must contain 'filename' column")
        
    X = df['filename'].values
    y = df.iloc[:, 1:].values
    
    test_size = (1-train_ratio)/2
    val_size = test_size
    
    # First split off test set (10%)
    msss_first = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,  # 10% for test
        random_state=random_seed
    )
    temp_idx, test_idx = next(msss_first.split(X, y))
    
    # Create temporary and test dataframes
    temp_df = df.iloc[temp_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    # Split remaining 90% into train (8/9) and val (1/9)
    msss_second = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=1/9,  # This gives us 80/10 split of remaining data
        random_state=random_seed
    )
    
    train_idx, val_idx = next(msss_second.split(
        temp_df['filename'].values,
        temp_df.iloc[:, 1:].values
    ))
    
    # Create final dataframes
    train_df = temp_df.iloc[train_idx].copy()
    val_df = temp_df.iloc[val_idx].copy()
    
    # Balance only training set
    train_df = balance_training_set(train_df, yolo_target)
    
    # Verify final split sizes
    total = len(train_df) + len(val_df) + len(test_df)
    logging.info("\nFinal split ratios:")
    logging.info(f"Train: {len(train_df)/total:.2%}")
    logging.info(f"Val: {len(val_df)/total:.2%}")
    logging.info(f"Test: {len(test_df)/total:.2%}")

    return (train_df['filename'].tolist(),
            val_df['filename'].tolist(),
            test_df['filename'].tolist(),
            train_df, val_df, test_df)
 
def log_all_split_distributions(train_df, val_df, test_df, yolo_target):
    """Log detailed distribution of images across splits and save to file.
    
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
    all_categories = get_target_columns(yolo_target)
    
    # Prepare distribution information
    distribution_info = []
    distribution_info.append("\nImage Distribution by Category:")
    distribution_info.append(f"{'Category':<20} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    distribution_info.append("-" * 55)

    for cat_name in all_categories:
        train_count = train_df[cat_name].sum()
        val_count = val_df[cat_name].sum()
        test_count = test_df[cat_name].sum()
        total = train_count + val_count + test_count
        line = f"{cat_name:<20} {train_count:<8} {val_count:<8} {test_count:<8} {total:<8}"
        distribution_info.append(line)
        # Also log to console
        logging.info(line)
    
    total_images = len(train_df) + len(val_df) + len(test_df)
    split_info = [
        f"\nTotal number of images: {total_images}",
        f"Train: {len(train_df)} ({len(train_df)/total_images:.2%})",
        f"Val: {len(val_df)} ({len(val_df)/total_images:.2%})",
        f"Test: {len(test_df)} ({len(test_df)/total_images:.2%})"
    ]
    distribution_info.extend(split_info)
    
    # Log to console
    for line in split_info:
        logging.info(line)
    
    # Save to file
    output_dir = Path(BasePaths.output_dir/"dataset_statistics")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"distribution_{yolo_target}_{timestamp}.txt"
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(distribution_info))
    
    logging.info(f"\nDistribution saved to: {output_file}")

def get_target_columns(yolo_target: str) -> list:
    """
    Returns list of target columns based on yolo_target type.
    
    Parameters:
    ----------
    yolo_target: str
        The target type for YOLO (e.g., "adult_person_face", "child_person_face", etc.)
    
    Returns:
    -------
    list
        List of column names that contain the target labels
    """
    target_columns_map = {
        "adult_person_face": ["adult_person", "adult_face"],
        "child_person_face": ["child_person", "child_face", "child_body_parts"],
        "object": ["book", "toy", "kitchenware", "screen", "other_object"],
        "person_face": ["person", "face"],
        "person_face_object": ["person", "face", "book", "toy", "kitchenware", 
                             "screen", "other_object"],
        "all": ["adult_person", "child_person", "adult_face", "child_face",
                "book", "toy", "kitchenware", "screen", "other_object"]
    }
    
    if yolo_target not in target_columns_map:
        raise ValueError(f"Invalid yolo_target: {yolo_target}")
        
    return target_columns_map[yolo_target]
         
def move_images(yolo_target: str, 
                image_names: list, 
                split_type: str, 
                label_path: Path,
                n_workers: int = 4) -> Tuple[int, int]:
    """
    Move images and their corresponding labels to the specified split directory.
    Uses multithreading for faster processing.
    
    Parameters
    ----------
    yolo_target: str
        Target type for YOLO (e.g., "person_face", "person_face_object", "gaze")
    image_names: list
        List of image names to process
    split_type: str
        Split type (train, val, or test)
    label_path: Path
        Path to label directory
    n_workers: int
        Number of worker threads for parallel processing
        
    Returns
    -------
    Tuple[int, int]
        Number of successful and failed moves
    
    Raises
    ------
    ValueError
        If yolo_target is invalid or paths cannot be determined
    """
    if not image_names:
        logging.info(f"No images to move for {yolo_target} {split_type}")
        return (0, 0)

    # Get destination paths
    paths = (ResNetPaths.get_target_paths(yolo_target, split_type) 
            if yolo_target in ["child_face", "adult_face", "adult_person", "child_person"]
            else YoloPaths.get_target_paths(yolo_target, split_type))
    
    if not paths:
        raise ValueError(f"Invalid yolo_target: {yolo_target}")
    
    image_dst_dir, label_dst_dir = paths
    image_dst_dir.mkdir(parents=True, exist_ok=True)
    label_dst_dir.mkdir(parents=True, exist_ok=True)

    # Define source directory mapping
    input_dir_mapping = {
        "gaze": DetectionPaths.gaze_images_input_dir,
        "no_gaze": DetectionPaths.gaze_images_input_dir,
        "child_face": DetectionPaths.face_images_input_dir,
        "adult_face": DetectionPaths.face_images_input_dir,
        "adult_person": DetectionPaths.person_images_input_dir,
        "child_person": DetectionPaths.person_images_input_dir
    }

    def process_single_image(image_name: str) -> bool:
        """Process a single image and its label."""
        try:
            if yolo_target in ["all", "child_person_face", "adult_person_face", 
                             "object", "person_face", "person_face_object"]:
                # Handle detection cases
                image_parts = image_name.split("_")[:8]
                image_folder = "_".join(image_parts)
                image_src = DetectionPaths.images_input_dir / image_folder / f"{image_name}.jpg"
                label_src = label_path / f"{image_name}.txt"
                image_dst = image_dst_dir / f"{image_name}.jpg"
                label_dst = label_dst_dir / f"{image_name}.txt"

                # Handle label file
                if not label_src.exists():
                    label_dst.touch()
                else:
                    shutil.copy2(label_src, label_dst)

                # Handle image file
                if not image_src.exists():
                    logging.debug(f"Image not found: {image_src}")
                    return False
                shutil.copy2(image_src, image_dst)

            else:
                # Handle classification cases
                input_dir = input_dir_mapping.get(yolo_target)
                if not input_dir:
                    logging.error(f"No input directory for target: {yolo_target}")
                    return False

                image_src = input_dir / image_name
                image_dst = image_dst_dir / image_name

                if not image_src.exists():
                    logging.debug(f"Image not found: {image_src}")
                    return False
                shutil.copy2(image_src, image_dst)

            return True

        except Exception as e:
            logging.error(f"Error processing {image_name}: {str(e)}")
            return False

    # Process images in parallel with progress bar
    successful = failed = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_single_image, img) for img in image_names]
        
        with tqdm(total=len(image_names), desc=f"Moving {split_type} images") as pbar:
            for future in as_completed(futures):
                if future.result():
                    successful += 1
                else:
                    failed += 1
                pbar.update(1)

    # Log results
    logging.info(f"\nCompleted moving {split_type} images:")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")
    
    return successful, failed

def extract_id(filename):
    """Extract the ID from a filename (e.g., 'id255237' from 'quantex_at_home_id255237_...')."""
    parts = filename.split('_')
    for part in parts:
        if part.startswith('id'):
            return part
    return None

def get_original_image_and_index(face_image):
    """Get the original image name and detection index from a face image filename."""
    base_name = face_image.split('_face_')[0]
    index = int(face_image.split('_face_')[1].split('.')[0])
    original_image = base_name + '.jpg'
    return original_image, index

def get_class(face_image, annotation_folder):
    """Retrieve the class ID (0 or 1) for a face image from its annotation file."""
    original_image, index = get_original_image_and_index(face_image)
    annotation_file = annotation_folder / Path(original_image).with_suffix('.txt').name
    if annotation_file.exists():
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
            if index < len(lines):
                class_id = lines[index].strip().split()[0]
                return int(class_id)
    return None

def compute_id_counts(face_images, annotation_folder):
    """Compute the number of class 0 and class 1 face images per ID."""
    id_counts = defaultdict(lambda: [0, 0])  # [n0, n1] for each ID
    for face_image in face_images:
        id_ = extract_id(face_image)
        class_id = get_class(face_image, annotation_folder)
        if class_id == 0:
            id_counts[id_][0] += 1
        elif class_id == 1:
            id_counts[id_][1] += 1
    return id_counts

def find_best_split(all_ids, id_counts, total_samples, num_trials=100):
    """Find the best split of IDs that maintains class distribution in val and test sets."""
    best_score = float('inf')
    best_split = None
    overall_ratio = sum(counts[0] for counts in id_counts.values()) / total_samples

    for _ in range(num_trials):
        random.shuffle(all_ids)
        # Cumulative sum of samples
        cumsums = [0]
        for id_ in all_ids:
            cumsums.append(cumsums[-1] + id_counts[id_][0] + id_counts[id_][1])
        
        # Find the split point for validation
        target_val = 0.1 * total_samples
        k = min(range(1, len(cumsums)), key=lambda i: abs(cumsums[i] - target_val))
        val_ids = all_ids[:k]
        
        remaining_ids = all_ids[k:]
        cumsums_remaining = [0]
        for id_ in remaining_ids:
            cumsums_remaining.append(cumsums_remaining[-1] + id_counts[id_][0] + id_counts[id_][1])
        
        # Find the split point for test
        target_test = 0.1 * total_samples
        m = min(range(1, len(cumsums_remaining)), key=lambda i: abs(cumsums_remaining[i] - target_test))
        test_ids = remaining_ids[:m]
        train_ids = remaining_ids[m:]

        # Compute class distribution in val and test
        n0_val = sum(id_counts[id_][0] for id_ in val_ids)
        n1_val = sum(id_counts[id_][1] for id_ in val_ids)
        n_val = n0_val + n1_val
        
        n0_test = sum(id_counts[id_][0] for id_ in test_ids)
        n1_test = sum(id_counts[id_][1] for id_ in test_ids)
        n_test = n0_test + n1_test

        # Calculate how close the class ratios are to the overall ratio
        if n_val > 0 and n_test > 0:
            val_ratio = n0_val / n_val
            test_ratio = n0_test / n_test
            score = abs(val_ratio - overall_ratio) + abs(test_ratio - overall_ratio)
            if score < best_score:
                best_score = score
                best_split = (val_ids, test_ids, train_ids)
    
    return best_split

def balance_train_set(train_face_images, annotation_folder):
    """Balance the training set to a 50/50 class ratio by dropping majority class samples."""
    train_class_0 = [f for f in train_face_images if get_class(f, annotation_folder) == 0]
    train_class_1 = [f for f in train_face_images if get_class(f, annotation_folder) == 1]
    
    if len(train_class_0) > len(train_class_1):
        train_class_0_balanced = random.sample(train_class_0, len(train_class_1))
        return train_class_0_balanced + train_class_1
    else:
        train_class_1_balanced = random.sample(train_class_1, len(train_class_0))
        return train_class_0 + train_class_1_balanced

def split_dataset(face_folder, annotation_folder):
    """Split the dataset into train, val, and test sets with the specified conditions."""
    # Get all face images
    face_folder = Path(face_folder)
    annotation_folder = Path(annotation_folder)
    all_face_images = [f for f in os.listdir(face_folder) if f.endswith('.jpg')]
    
    logging.info(f"Found {len(all_face_images)} face images in {face_folder}")
    # Compute class counts per ID
    id_counts = compute_id_counts(all_face_images, annotation_folder)
    all_ids = list(id_counts.keys())
    
    # Get total distribution
    N0 = sum(counts[0] for counts in id_counts.values())
    N1 = sum(counts[1] for counts in id_counts.values())
    total_samples = N0 + N1
    logging.info(f"Class 0 (Child Faces): {N0} images")
    logging.info(f"Class 1 (Adult Faces): {N1} images")
    logging.info(f"Total: {total_samples} images")
    logging.info(f"Overall Child-to-Total Ratio: {N0 / total_samples:.3f}")

    # Find the best split
    val_ids, test_ids, train_ids = find_best_split(all_ids, id_counts, total_samples)
    
    # Assign face images to splits
    val_face_images = [f for f in all_face_images if extract_id(f) in val_ids]
    test_face_images = [f for f in all_face_images if extract_id(f) in test_ids]
    train_face_images = [f for f in all_face_images if extract_id(f) in train_ids]
    
    # Balance the training set
    train_balanced = balance_train_set(train_face_images, annotation_folder)
    
    # Log the results
    for split_name, split_images in [("Validation", val_face_images), 
                                     ("Test", test_face_images), 
                                     ("Train (Balanced)", train_balanced)]:
        n0 = sum(1 for f in split_images if get_class(f, annotation_folder) == 0)
        n_split = len(split_images)
        logging.info(f"{split_name} Set: {n_split} images, Child-to-Total Ratio: {n0 / n_split if n_split > 0 else 0:.3f}")
    
    return train_balanced, val_face_images, test_face_images
           
def split_yolo_data(label_path: Path, yolo_target: str):
    """
    This function prepares the dataset for YOLO training by splitting the images into train, val, and test sets.
    
    Parameters:
    ----------
    label_path: Path
        Path to the directory containing label files.
    yolo_target: str
        The target object for YOLO detection or classification.
    """
    logging.info(f"Starting dataset preparation for {yolo_target}")
    total_images = get_total_number_of_annotated_frames(label_path)

    try:
        if yolo_target == "face":
            # Define source directories
            face_folder = DetectionPaths.face_images_input_dir
            annotation_folder = label_path

            # Get custom splits (assumes split_dataset is defined elsewhere)
            train_images, val_images, test_images = split_dataset(face_folder, annotation_folder)

            # Process each split
            for split_name, split_images in [("train", train_images), ("val", val_images), ("test", test_images)]:
                # Separate images into child and adult classes
                child_images = [img for img in split_images if get_class(img, annotation_folder) == 0]
                adult_images = [img for img in split_images if get_class(img, annotation_folder) == 1]

                # Move child face images
                if child_images:
                    successful, failed = move_images(
                        yolo_target="child_face",
                        image_names=child_images,
                        split_type=split_name,
                        label_path=label_path,
                        n_workers=4
                    )
                    logging.info(f"{split_name} child_face: Moved {successful} images, Failed {failed}")
                else:
                    logging.warning(f"No child_face images for {split_name}")

                # Move adult face images
                if adult_images:
                    successful, failed = move_images(
                        yolo_target="adult_face",
                        image_names=adult_images,
                        split_type=split_name,
                        label_path=label_path,
                        n_workers=4
                    )
                    logging.info(f"{split_name} adult_face: Moved {successful} images, Failed {failed}")
                else:
                    logging.warning(f"No adult_face images for {split_name}")

        elif yolo_target in ["person", "gaze"]:
            # Handle other binary classification cases (unchanged)
            class_mapping = {
                "person": {0: "child_person", 1: "adult_person"},
                "gaze": {0: "no_gaze", 1: "gaze"}
            }
            images_class_0, images_class_1 = get_binary_class_distribution(
                total_images, label_path, yolo_target
            )
            
            splits_dict = binary_stratified_split(
                (images_class_0, images_class_1), yolo_target,
                test_size=0.1, val_size=0.1, random_state=TrainingConfig.random_seed
            )
            
            for class_idx, binary_class in class_mapping[yolo_target].items():
                train, val, test = splits_dict[class_idx]
                logging.info(f"\nProcessing {binary_class}:")
                
                for split_name, split_set in [("train", train), ("val", val), ("test", test)]:
                    if split_set:
                        successful, failed = move_images(
                            yolo_target=binary_class,
                            image_names=split_set,
                            split_type=split_name,
                            label_path=label_path,
                            n_workers=4
                        )
                        logging.info(f"{split_name}: Moved {successful} images, Failed {failed}")
                    else:
                        logging.warning(f"No images for {binary_class} {split_name} split")
                        
        else:
            # Handle multi-class cases (unchanged)
            df = get_class_distribution(total_images, label_path, yolo_target)
            
            train, val, test, train_df, val_df, test_df = multilabel_stratified_split(
                df, yolo_target=yolo_target
            )
            
            log_all_split_distributions(train_df, val_df, test_df, yolo_target)
            
            total_successful = total_failed = 0
            for split_name, split_set in [("train", train), ("val", val), ("test", test)]:
                if split_set:
                    successful, failed = move_images(
                        yolo_target=yolo_target,
                        image_names=split_set,
                        split_type=split_name,
                        label_path=label_path,
                        n_workers=4
                    )
                    total_successful += successful
                    total_failed += failed
                    logging.info(f"{split_name}: Moved {successful} images, Failed {failed}")
                else:
                    logging.warning(f"No images for {split_name} split")
            
            logging.info(f"\nTotal images processed: {total_successful + total_failed}")
            logging.info(f"Successfully moved: {total_successful}")
            logging.info(f"Failed to move: {total_failed}")
    except Exception as e:
        logging.error(f"Error processing target {yolo_target}: {str(e)}")
        raise
    
    logging.info(f"\nCompleted dataset preparation for {yolo_target}")
    
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
            "person_face": YoloPaths.person_face_labels_input_dir,
            "person_face_object": YoloPaths.person_face_object_labels_input_dir,
            "child_person_face": YoloPaths.child_person_face_labels_input_dir,
            "adult_person_face": YoloPaths.adult_person_face_labels_input_dir,
            "object": YoloPaths.object_labels_input_dir,
            "person": ResNetPaths.person_labels_input_dir,
            "face": ResNetPaths.face_labels_input_dir,
            "gaze": YoloPaths.gaze_labels_input_dir
        }
        label_path = path_mapping[yolo_target]
        image_folder = DetectionPaths.images_input_dir
        #train_images, val_images, test_images = split_dataset(image_folder, label_path)
        split_yolo_data(label_path, yolo_target)
        logging.info("Dataset preparation for YOLO completed.")
    elif model_target == "other_model":
        pass
    else:
        logging.error("Unsupported model target specified!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for model training.")
    parser.add_argument("--model_target", choices=["yolo", "other_model"], required=True, help="Specify the model type")
    parser.add_argument("--yolo_target", choices=["all", "gaze", "adult_person_face", "child_person_face", "object", "person_face", "person_face_object"], required=True, help="Specify the YOLO target type")
    
    args = parser.parse_args()
    main(args.model_target, args.yolo_target)
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
from constants import DetectionPaths, BasePaths, ClassificationPaths
from config import TrainingConfig
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit as MSSS
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Optional, Tuple
from sklearn.model_selection import StratifiedShuffleSplit

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        the total number of images along with their IDs
    """
    video_names = set()
    total_images = []
    
    # Step 1: Get unique video names
    for annotation_file in label_path.glob('*.txt'):
        parts = annotation_file.stem.split('_')
        video_name = "_".join(parts[:8])
        video_names.add(video_name)      
    
    logging.info(f"Found {len(video_names)} unique video names")

    
    # Step 2: Count total images and extract IDs
    for video_name in video_names:
        video_path = image_folder / video_name
        if video_path.exists() and video_path.is_dir():
            for video_file in video_path.iterdir():
                if video_file.is_file():
                    image_name = video_file.name
                    # extract image_id from file_name 
                    parts = image_name.split("_")
                    image_id = parts[3].replace("id", "")
                    
                    total_images.append((str(video_file.resolve()), image_id))
    return total_images   


def get_class_distribution(total_images: list, annotation_folder: Path, target_type: str) -> pd.DataFrame:
    """
    Reads label files and groups images based on their class distribution.

    Parameters:
    ----------
    total_images: list
        List of tuples containing image paths and IDs.
    annotation_folder: Path
        Path to the directory containing label files.
    target_type: str
        The target type for classification (e.g., "person_face_object")

    Returns:
    -------
    pd.DataFrame
        DataFrame containing image filenames, IDs, and their corresponding one-hot encoded class labels.
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

    for image_path, image_id in total_images:
        image_file = Path(image_path)
        annotation_file = annotation_folder / image_file.with_suffix('.txt').name

        # Get labels from annotation file
        labels = []
        if annotation_file.exists() and annotation_file.stat().st_size > 0:
            with open(annotation_file, 'r') as f:
                class_ids = {int(line.split()[0]) for line in f if line.split()}
                labels = [id_to_name[cid] for cid in class_ids if cid in id_to_name]

        # Create one-hot encoded dictionary for the image
        image_class_mapping.append({
            "filename": image_file.stem,
            "id": image_id,
            "has_annotation": bool(labels),  # True if labels are found, False otherwise
            **{class_name: (1 if class_name in labels else 0) 
            for class_name in id_to_name.values()}
        })

    return pd.DataFrame(image_class_mapping)

def balance_training_set(df: pd.DataFrame) -> pd.DataFrame:
    """Helper function to balance training set based on existing has_annotation column.
    
    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing image filenames and their corresponding IDs.
        Must have a 'has_annotation' column.
        
    Returns
    -------
    pd.DataFrame
        Balanced DataFrame with equal numbers of positive and negative samples.
    """
    pos_samples = df[df['has_annotation']].copy()
    neg_samples = df[~df['has_annotation']].copy()
    
    # Balance negative samples to match positive
    if len(neg_samples) > len(pos_samples):
        neg_samples = neg_samples.sample(
            n=len(pos_samples),
            random_state=TrainingConfig.random_seed
        )
    
    return pd.concat([pos_samples, neg_samples]).drop('has_annotation', axis=1)
     
def multilabel_stratified_split(df: pd.DataFrame,
                                train_ratio: float = TrainingConfig.train_test_split_ratio,
                                random_seed: int = TrainingConfig.random_seed):
    """
    Performs stratified split maintaining 80/10/10 ratio.
    Only balances training set to preserve real-world distribution in val/test.
    Ensures images from the same ID are in the same split.
    
    Parameters
    -----------
    df: pd.DataFrame
        DataFrame containing image filenames and their corresponding IDs.
    train_ratio: float
        Ratio of training data (default is 0.8).
    random_seed: int
        Random seed for reproducibility (default is 42).
    """
    if df.empty:
        raise ValueError("Empty DataFrame provided")
    if 'filename' not in df.columns or 'id' not in df.columns:
        raise ValueError("DataFrame must contain 'filename' and 'id' columns")
        
    # Group by ID and shuffle IDs
    ids = df['id'].unique()
    np.random.seed(random_seed)
    np.random.shuffle(ids)
    
    train_ids = ids[:int(len(ids) * train_ratio)]
    val_ids = ids[int(len(ids) * train_ratio):int(len(ids) * (train_ratio + (1-train_ratio)/2))]
    test_ids = ids[int(len(ids) * (train_ratio + (1-train_ratio)/2)):]

    # Split data based on IDs
    train_df = df[df['id'].isin(train_ids)].copy()
    val_df = df[df['id'].isin(val_ids)].copy()
    test_df = df[df['id'].isin(test_ids)].copy()

    # Balance only training set
    train_df = balance_training_set(train_df)
    
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
    paths = (ClassificationPaths.get_target_paths(yolo_target, split_type) 
            if yolo_target in ["child_face", "adult_face", "adult_person", "child_person", "gaze", "no_gaze"]
            else DetectionPaths.get_target_paths(yolo_target, split_type))
    
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

def get_original_image_and_index(input_image: str) -> Tuple[str, int]:
    """Extract the original image name and detection index from an input image filename.
    
    Works for both "_face_" and "_person_" formats by always extracting the last `_`-separated number.

    Parameters
    ----------
    input_image: str
        The name of the input image file.
    
    Returns
    -------
    Tuple[str, int]
        The original image name and the detection index.
    """
    parts = input_image.rsplit('_', 2)  # Split at most twice from the right
    base_name = parts[0]  # Everything before the last two `_` parts
    index = int(parts[-1].split('.')[0])  # Extract last numeric part (index)
    
    original_image = base_name + '.jpg'
    return original_image, index

def get_class(input_image: str, annotation_folder: Path) -> Optional[int]:
    """Retrieve the class ID (0 or 1) for an input image from its annotation file.
    
    Parameters
    ----------
    input_image: str
        The name of the input image file.
    annotation_folder: Path
        Path to the directory containing annotation files.
        
    Returns
    -------
    Optional[int]
        The class ID (0 or 1) if found, otherwise None.
    """
    
    original_image, index = get_original_image_and_index(input_image)
    annotation_file = annotation_folder / Path(original_image).with_suffix('.txt').name
    if annotation_file.exists():
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
            if index < len(lines):
                class_id = lines[index].strip().split()[0]
                return int(class_id)
    return None

def compute_id_counts(input_images: list,
                      annotation_folder: Path) -> Tuple[defaultdict, int]:
    """Compute the number of images per ID and their class distribution.
    
    Parameters
    ----------
    input_images: list
        List of image filenames.
    annotation_folder: Path
        Path to the directory containing annotation files.
    
    Returns
    -------
    Tuple[defaultdict, int]
        A dictionary with IDs as keys and a list of counts [n0, n1] as values,
        where n0 is the count of class 0 and n1 is the count of class 1.
        Also returns the number of images without annotations.
    """
    id_counts = defaultdict(lambda: [0, 0])  # [n0, n1] for each ID
    missing_annotations = 0
    missing_images = []
    for input_image in input_images:
        id_ = extract_id(input_image)
        class_id = get_class(input_image, annotation_folder)
        if class_id == 0:
            id_counts[id_][0] += 1
        elif class_id == 1:
            id_counts[id_][1] += 1
        else:
            missing_annotations += 1
            missing_images.append(input_image)
    logging.info(f"Images without annotations: {missing_annotations}")
    return id_counts, missing_annotations

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

def balance_train_set(train_input_images: list, 
                      annotation_folder: Path,
                      min_ratio: float = 0.45) -> list:
    """
    Balance the training set only if class ratio exceeds specified threshold.
    
    Parameters
    ----------
    train_input_images: list
        List of training image filenames.
    annotation_folder: Path
        Path to the directory containing annotation files.
    min_ratio: float
        Minimum ratio of minority class to total images to trigger balancing.
        
    Returns
    -------
    list
        A balanced list of training image filenames.
    """
    # Separate images by class
    train_class_0 = []
    train_class_1 = []
    
    for img in train_input_images:
        class_id = get_class(img, annotation_folder)
        if class_id == 0:
            train_class_0.append(img)
        elif class_id == 1:
            train_class_1.append(img)

    n0 = len(train_class_0)
    n1 = len(train_class_1)
    total = n0 + n1
    
    # Calculate class ratios
    ratio_0 = n0 / total if total > 0 else 0
    ratio_1 = n1 / total if total > 0 else 0
    
        
    # Only balance if ratio exceeds threshold
    if min(ratio_0, ratio_1) < min_ratio:
        logging.info("Class imbalance detected, performing balancing...")
        if len(train_class_0) > len(train_class_1):
            target_size = len(train_class_1)
            train_class_0 = random.sample(train_class_0, target_size)
        else:
            target_size = len(train_class_0)
            train_class_1 = random.sample(train_class_1, target_size)
    
        balanced_ratio = len(train_class_0) / (len(train_class_0) + len(train_class_1))
        logging.info(f"After balancing - Class ratio: {balanced_ratio:.3f}")
        return train_class_0 + train_class_1
    else:
        logging.info("Class distribution within acceptable range, no balancing needed")
        return train_input_images
    
def split_dataset(input_folder: str, 
                  annotation_folder: str,
                  yolo_target: str,
                  class_mapping: dict = None) -> Tuple[list, list, list]:
    """
    Split the dataset into train, val, and test sets while 
    
    Parameters
    ----------
    input_folder: str
        Path to the folder containing input images.
    annotation_folder: str
        Path to the folder containing annotation files.
    yolo_target: str
        The target type for YOLO detection or classification.
    class_mapping: dict
        Optional mapping of class IDs to names.
    """
    # Get all input images
    input_folder = Path(input_folder)
    annotation_folder = Path(annotation_folder)
    all_input_images = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]
    
    # Create output directory for split information
    output_dir = Path(BasePaths.output_dir/"dataset_statistics")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"split_distribution_{yolo_target}_{timestamp}.txt"
    
    split_info = []
    split_info.append(f"Dataset Split Information - {timestamp}\n")
    split_info.append(f"Found {len(all_input_images)} {yolo_target} images in {input_folder}\n")
    
    # Compute class counts per ID
    id_counts, missing_annotations = compute_id_counts(all_input_images, annotation_folder)
    all_ids = list(id_counts.keys())
    
    # Get total distribution
    N0 = sum(counts[0] for counts in id_counts.values())
    N1 = sum(counts[1] for counts in id_counts.values())
    total_samples = N0 + N1
    
    # Log initial distribution
    split_info.append("Initial Distribution:")
    split_info.append(f"Class 0 {class_mapping[0][0]}: {N0} images")
    split_info.append(f"Class 1 {class_mapping[1][0]}: {N1} images")
    split_info.append(f"Total: {total_samples} images")
    split_info.append(f"Missing Annotations: {missing_annotations} images\n")
    split_info.append(f"Overall {class_mapping[0][0]}-to-Total Ratio: {N0 / total_samples:.3f}\n")

    # Find the best split
    val_ids, test_ids, train_ids = find_best_split(all_ids, id_counts, total_samples)
    
    # Assign input images to splits
    val_input_images = [f for f in all_input_images if extract_id(f) in val_ids]
    test_input_images = [f for f in all_input_images if extract_id(f) in test_ids]
    train_input_images = [f for f in all_input_images if extract_id(f) in train_ids]
    
    # Balance the training set
    train_balanced = balance_train_set(train_input_images, annotation_folder)
    
    # Log detailed split information
    split_info.append("Split Distribution:")
    split_info.append("-" * 50)
    
    for split_name, split_images in [
        ("Validation", val_input_images), 
        ("Test", test_input_images), 
        ("Train (Original)", train_input_images),
        ("Train (Balanced)", train_balanced)
    ]:
        n0 = sum(1 for f in split_images if get_class(f, annotation_folder) == 0)
        n1 = len(split_images) - n0
        n_split = len(split_images)
        ratio = n0 / n_split if n_split > 0 else 0
        
        split_details = [
            f"\n{split_name} Set:",
            f"Total Images: {n_split}",
            f"{class_mapping[0][0]} (Class 0): {n0}",
            f"{class_mapping[1][0]} (Class 1): {n1}",
            f"{class_mapping[0][0]}-to-Total Ratio: {ratio:.3f}"
        ]
        split_info.extend(split_details)
        
        # Also log to console
        logging.info("\n".join(split_details))
    
    # Add ID distribution information
    split_info.extend([
        f"\nID Distribution:",
        f"Training IDs: {len(train_ids)}, {train_ids}",
        f"Validation IDs: {len(val_ids)}, {val_ids}",
        f"Test IDs: {len(test_ids)}, {test_ids}",
    ])
    
    # Check for ID overlap
    train_id_set = set(train_ids)
    val_id_set = set(val_ids)
    test_id_set = set(test_ids)
    
    overlap = train_id_set & val_id_set | train_id_set & test_id_set | val_id_set & test_id_set
    split_info.append(f"\nID Overlap Check:")
    split_info.append(f"Overlap found: {'Yes' if overlap else 'No'}")
    if overlap:
        split_info.append(f"Overlapping IDs: {overlap}")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(split_info))
    
    logging.info(f"\nSplit distribution saved to: {output_file}")
    
    return train_balanced, val_input_images, test_input_images
           
def split_yolo_data(annotation_folder: Path, yolo_target: str):
    """
    This function prepares the dataset for YOLO training by splitting the images into train, val, and test sets.
    
    Parameters:
    ----------
    annotation_folder: Path
        Path to the directory containing label files.
    yolo_target: str
        The target object for YOLO detection or classification.
    """
    logging.info(f"Starting dataset preparation for {yolo_target}")

    try:
        # Define mappings for different target types
        target_mappings = {
            "face": {
                1: ("adult_face", DetectionPaths.face_images_input_dir),
                0: ("child_face", DetectionPaths.face_images_input_dir)
            },
            "person": {
                1: ("adult_person", DetectionPaths.person_images_input_dir),
                0: ("child_person", DetectionPaths.person_images_input_dir)
            },
            "gaze": {
                0: ("no_gaze", DetectionPaths.gaze_images_input_dir),
                1: ("gaze", DetectionPaths.gaze_images_input_dir)
            }
        }       
         
        if yolo_target in target_mappings:
            # Get source directories based on target type
            input_folder = target_mappings[yolo_target][0][1]  # Use first mapping's input dir
            class_mapping = target_mappings[yolo_target]
            # Get custom splits
            train_images, val_images, test_images = split_dataset(input_folder, annotation_folder, yolo_target, class_mapping)

            # Process each split
            for split_name, split_images in [("train", train_images), ("val", val_images), ("test", test_images)]:
                # Separate images into classes
                class_0_images = [img for img in split_images if get_class(img, annotation_folder) == 0]
                class_1_images = [img for img in split_images if get_class(img, annotation_folder) == 1]

                # Move class 0 images
                if class_0_images:
                    successful, failed = move_images(
                        yolo_target=target_mappings[yolo_target][0][0],
                        image_names=class_0_images,
                        split_type=split_name,
                        label_path=annotation_folder,
                        n_workers=4
                    )
                    logging.info(f"{split_name} {target_mappings[yolo_target][0][0]}: Moved {successful} images, Failed {failed}")
                else:
                    logging.warning(f"No {target_mappings[yolo_target][0][0]} images for {split_name}")

                # Move class 1 images
                if class_1_images:
                    successful, failed = move_images(
                        yolo_target=target_mappings[yolo_target][1][0],
                        image_names=class_1_images,
                        split_type=split_name,
                        label_path=annotation_folder,
                        n_workers=4
                    )
                    logging.info(f"{split_name} {target_mappings[yolo_target][1][0]}: Moved {successful} images, Failed {failed}")
                else:
                    logging.warning(f"No {target_mappings[yolo_target][1][0]} images for {split_name}")
                        
        else:
            # count number of total images in annotated videos
            total_images = get_total_number_of_annotated_frames(annotation_folder)
            # get data distribution per frame
            df = get_class_distribution(total_images, annotation_folder, yolo_target)
            # split data grouped by id
            train, val, test, train_df, val_df, test_df = multilabel_stratified_split(df)
            
            log_all_split_distributions(train_df, val_df, test_df, yolo_target)
            
            total_successful = total_failed = 0
            for split_name, split_set in [("train", train), ("val", val), ("test", test)]:
                if split_set:
                    successful, failed = move_images(
                        yolo_target=yolo_target,
                        image_names=split_set,
                        split_type=split_name,
                        label_path=annotation_folder,
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
            "person_face": DetectionPaths.person_face_labels_input_dir,
            "person_face_object": DetectionPaths.person_face_object_labels_input_dir,
            "object": DetectionPaths.object_labels_input_dir,
            "person": ClassificationPaths.person_labels_input_dir,
            "face": ClassificationPaths.face_labels_input_dir,
            "gaze": ClassificationPaths.gaze_labels_input_dir
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
    parser.add_argument("--yolo_target", choices=["all", "gaze", "adult_person_face", "child_person_face", "object", "person_face", "person_face_object"], required=True, help="Specify the YOLO target type")
    
    args = parser.parse_args()
    main(args.model_target, args.yolo_target)
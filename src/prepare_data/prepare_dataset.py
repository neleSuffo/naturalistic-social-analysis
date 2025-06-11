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
from constants import DetectionPaths, BasePaths, ClassificationPaths, VALID_TARGETS
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
    This function returns only the frames that have annotations (every 30th frame).
    
    Parameters
    ----------
    label_path : Path
        the path to the label files
    image_folder : Path
        the path to the image folder
        
    Returns
    -------
    list
        the annotated frames (formatted as a list of tuples)
        e.g. [(image_path, image_id), ...]
    """
    video_names = set()
    total_images = []
    
    # Step 1: Get unique video names and annotated frames
    annotated_frames = set()
    for annotation_file in label_path.glob('*.txt'):
        parts = annotation_file.stem.split('_')
        video_name = "_".join(parts[:8])
        video_names.add(video_name)
        annotated_frames.add(annotation_file.stem)
    
    logging.info(f"Found {len(video_names)} unique video names")

    # Step 2: Only get the corresponding annotated frames from image folders
    for video_name in video_names:
        video_path = image_folder / video_name
        if video_path.exists() and video_path.is_dir():
            for video_file in video_path.iterdir():
                if video_file.is_file():
                    # Only include if the frame is in our annotated frames set
                    if video_file.stem in annotated_frames:
                        image_name = video_file.name
                        parts = image_name.split("_")
                        image_id = parts[3].replace("id", "")
                        total_images.append((str(video_file.resolve()), image_id))
                        
    logging.info(f"Total annotated frames found: {len(total_images)}")
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
        The target type for classification (e.g., "all")

    Returns:
    -------
    pd.DataFrame
        DataFrame containing image filenames, IDs, and their corresponding one-hot encoded class labels.
    """
    # Define class mappings based on target type
    class_mappings = {
        "person_face": {
            0: "person", 1: "face", 2: "child_body_parts"
        },
        "all": {
            0: "person", 1: "face", 2: "child_body_parts", 3: "object",
        }
        # "all": {
        #     0: "person", 1: "face", 2: "child_body_parts", 3: "book",
        #     4: "toy", 5: "kitchenware", 6: "screen", 7: "other_object", # map screen for now to other_object
        #     8: "other_object", 9: "other_object" # map former food and animal class to other_object
        # }
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
       
def multilabel_stratified_split(df: pd.DataFrame,
                              train_ratio: float = TrainingConfig.train_test_split_ratio,
                              random_seed: int = TrainingConfig.random_seed,
                              target: str = None):
    """
    Performs a group-based stratified split balancing three factors:
    1. Group Integrity: Keeps all frames from an ID in one split.
    2. Class Representation: Prioritizes ensuring all classes are present in val/test.
    3. True Frame Distribution: Bases the split ratio on the number of frames.
    
    Parameters:
    ----------
    df: pd.DataFrame
        DataFrame containing image filenames, IDs, and one-hot encoded class labels.
    train_ratio: float
        Target ratio for the training set based on frame count.
    val_ratio: float
        Target ratio for the validation set based on frame count.
    random_seed: int
        Random seed for reproducibility.
    target: str
        A name for the dataset, used for the output log file.
        
    Returns:
    -------
    Tuple[List[str], List[str], List[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Lists of filenames and DataFrames for train, val, and test splits.
    """
    val_ratio = (1.0 - train_ratio) / 2.0  # Ensure val and test are equal

    # --- 1. Setup and Pre-analysis ---
    output_dir = BasePaths.output_dir / "dataset_statistics"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"split_distribution_{target}_{timestamp}.txt"

    logging.info(f"Starting split for {df.shape[0]} frames and {df['id'].nunique()} unique IDs.")
    if df.empty:
        raise ValueError("Input DataFrame is empty!")

    class_columns = [col for col in df.columns if col not in ['filename', 'id', 'has_annotation']]
    id_class_map = df.groupby('id')[class_columns].sum().apply(lambda s: s[s > 0].index.to_list(), axis=1).to_dict()

    # Pre-calculate the 'weight' (frame count) of each ID
    id_frame_count_map = df['id'].value_counts().to_dict()
    total_frame_count = df.shape[0]

    # --- 2. Main Hybrid Scoring Algorithm ---
    available_ids = sorted(list(id_class_map.keys()))
    random.shuffle(available_ids) # Shuffle to prevent any bias from sorting order

    train_ids, val_ids, test_ids = set(), set(), set()
    
    # Track split sizes by cumulative frame count
    train_frames, val_frames, test_frames = 0, 0, 0
    
    # Track classes still needed in val and test
    uncovered_val_classes = set(class_columns)
    uncovered_test_classes = set(class_columns)

    # Define weights for the scoring model. Coverage bonus must be very high.
    COVERAGE_BONUS_WEIGHT = 1000.0
    RATIO_PENALTY_WEIGHT = 1.0 / total_frame_count # Normalize penalty by total frames

    for id_to_assign in available_ids:
        id_classes = set(id_class_map[id_to_assign])
        id_frame_count = id_frame_count_map[id_to_assign]
        
        scores = {'train': 0.0, 'val': 0.0, 'test': 0.0}

        # === Score for assigning to VALIDATION set ===
        # 1. Coverage Score: Huge bonus for covering a needed class
        val_coverage_gain = len(id_classes.intersection(uncovered_val_classes))
        scores['val'] += val_coverage_gain * COVERAGE_BONUS_WEIGHT
        
        # 2. Ratio Score: Penalty for making the split larger than its target frame count
        if val_frames >= total_frame_count * val_ratio:
            potential_new_val_frames = val_frames + id_frame_count
            overage_penalty = potential_new_val_frames - (total_frame_count * val_ratio)
            scores['val'] -= overage_penalty * RATIO_PENALTY_WEIGHT

        # === Score for assigning to TEST set ===
        # 1. Coverage Score
        test_coverage_gain = len(id_classes.intersection(uncovered_test_classes))
        scores['test'] += test_coverage_gain * COVERAGE_BONUS_WEIGHT

        # 2. Ratio Score
        test_ratio = 1.0 - train_ratio - val_ratio
        if test_frames >= total_frame_count * test_ratio:
            potential_new_test_frames = test_frames + id_frame_count
            overage_penalty = potential_new_test_frames - (total_frame_count * test_ratio)
            scores['test'] -= overage_penalty * RATIO_PENALTY_WEIGHT
            
        # === Assign ID to the split with the highest score ===
        best_split = max(scores, key=scores.get)
        
        if best_split == 'val':
            val_ids.add(id_to_assign)
            val_frames += id_frame_count
            uncovered_val_classes.difference_update(id_classes)
        elif best_split == 'test':
            test_ids.add(id_to_assign)
            test_frames += id_frame_count
            uncovered_test_classes.difference_update(id_classes)
        else: # best_split == 'train'
            train_ids.add(id_to_assign)
            train_frames += id_frame_count

    # --- 3. Finalization and Enhanced Logging ---
    train_df = df[df['id'].isin(train_ids)].copy()
    val_df = df[df['id'].isin(val_ids)].copy()
    test_df = df[df['id'].isin(test_ids)].copy()

    # Final check to warn about impossible splits
    for class_col in class_columns:
        if val_df[class_col].sum() == 0:
            logging.error(f"WARNING: Class '{class_col}' has 0 instances in the Validation set.")
        if test_df[class_col].sum() == 0:
            logging.error(f"WARNING: Class '{class_col}' has 0 instances in the Test set.")

    # Prepare detailed log report
    split_info = [f"Dataset Split Information - {timestamp}\n"]
    split_info.append(f"Total Frames: {total_frame_count}, Total IDs: {len(id_class_map)}\n")

    split_info.append("--- Split Distribution by FRAMES (Primary Goal) ---")
    split_info.append(f"{'Split':<12} {'Frames':<10} {'Percentage':<10}")
    split_info.append("-" * 40)
    split_info.append(f"{'Train':<12} {train_frames:<10} {train_frames/total_frame_count:<10.1%}")
    split_info.append(f"{'Validation':<12} {val_frames:<10} {val_frames/total_frame_count:<10.1%}")
    split_info.append(f"{'Test':<12} {test_frames:<10} {test_frames/total_frame_count:<10.1%}")
    
    split_info.append("\n--- Split Distribution by IDs (For Information) ---")
    split_info.append(f"{'Split':<12} {'IDs':<10} {'Percentage':<10}")
    split_info.append("-" * 40)
    split_info.append(f"{'Train':<12} {len(train_ids):<10} {len(train_ids)/len(id_class_map):<10.1%}")
    split_info.append(f"{'Validation':<12} {len(val_ids):<10} {len(val_ids)/len(id_class_map):<10.1%}")
    split_info.append(f"{'Test':<12} {len(test_ids):<10} {len(test_ids)/len(id_class_map):<10.1%}")
    
    split_info.append("\n--- Class Distribution (by annotation instances) ---")
    header = f"{'Category':<20} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}"
    split_info.append(header)
    split_info.append("-" * 65)
    
    for col in class_columns:
        train_count = train_df[col].sum()
        val_count = val_df[col].sum()
        test_count = test_df[col].sum()
        total = train_count + val_count + test_count
        row = (f"{col:<20} {int(train_count):<10} {int(val_count):<10} "
               f"{int(test_count):<10} {int(total):<10}")
        split_info.append(row)
    
    # Write report to file and log to console
    report_string = '\n'.join(split_info)
    with open(output_file, 'w') as f:
        f.write(report_string)
    
    logging.info(f"\nSplit distribution report saved to: {output_file}\n")
    logging.info(report_string)
    
    return (train_df['filename'].tolist(), val_df['filename'].tolist(), test_df['filename'].tolist(),
            train_df, val_df, test_df)
         
def move_images(target: str, 
                image_names: list, 
                split_type: str, 
                label_path: Path,
                n_workers: int = 4) -> Tuple[int, int]:
    """
    Move images and their corresponding labels to the specified split directory.
    Uses multithreading for faster processing.
    
    Parameters
    ----------
    target: str
        Target type for YOLO
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
        If target is invalid or paths cannot be determined
    """
    if not image_names:
        logging.info(f"No images to move for {target} {split_type}")
        return (0, 0)

    # Get destination paths
    paths = (ClassificationPaths.get_target_paths(target, split_type) 
            if target in ["child_face", "adult_face", "adult_person", "child_person", "gaze", "no_gaze"]
            else DetectionPaths.get_target_paths(target, split_type))
    
    if not paths:
        raise ValueError(f"Invalid target: {target}")
    
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
            if target in ["person_face", "all"]:
                # Handle detection cases
                image_parts = image_name.split("_")[:8]
                image_folder = "_".join(image_parts)
                
                # Extract frame number from filename
                frame_num = int(image_name.split("_")[-1].split(".")[0])
                
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
                # Extract frame number from filename
                frame_num = int(image_name.split("_")[-1].split(".")[0])
                
                input_dir = input_dir_mapping.get(target)
                if not input_dir:
                    logging.error(f"No input directory for target: {target}")
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

def find_best_split(all_ids, id_counts, total_samples, num_trials=100, min_ratio=0.05):
    """Find the best split of IDs that maintains class distribution in val and test sets."""
    best_score = float('inf')
    best_split = None
    total_class_0 = sum(counts[0] for counts in id_counts.values())
    total_class_1 = sum(counts[1] for counts in id_counts.values())
    min_class_0_needed = total_class_0 * min_ratio
    min_class_1_needed = total_class_1 * min_ratio

    for _ in range(num_trials):
        # Sort IDs by ratio of classes to try balancing first
        id_ratios = [(id_, counts[0]/(counts[0] + counts[1]) if (counts[0] + counts[1]) > 0 else 0) 
                     for id_, counts in id_counts.items()]
        sorted_ids = sorted(id_ratios, key=lambda x: abs(x[1] - 0.5))  # Sort by how balanced they are
        trial_ids = [id_ for id_, _ in sorted_ids]
        random.shuffle(trial_ids)  # Add some randomness while keeping generally balanced IDs first

        # Initialize trial splits
        val_ids = []
        test_ids = []
        val_class_0 = val_class_1 = test_class_0 = test_class_1 = 0

        # First fill validation set until minimum requirements met
        for id_ in trial_ids[:]:
            if (val_class_0 >= min_class_0_needed and 
                val_class_1 >= min_class_1_needed):
                break
                
            val_ids.append(id_)
            trial_ids.remove(id_)
            val_class_0 += id_counts[id_][0]
            val_class_1 += id_counts[id_][1]

        # Then fill test set until minimum requirements met
        for id_ in trial_ids[:]:
            if (test_class_0 >= min_class_0_needed and 
                test_class_1 >= min_class_1_needed):
                break
                
            test_ids.append(id_)
            trial_ids.remove(id_)
            test_class_0 += id_counts[id_][0]
            test_class_1 += id_counts[id_][1]

        # Remaining IDs go to train
        train_ids = trial_ids

        # Calculate how well balanced the splits are
        val_total = val_class_0 + val_class_1
        test_total = test_class_0 + test_class_1
        
        if val_total > 0 and test_total > 0:
            val_ratio = val_class_0 / val_total
            test_ratio = test_class_0 / test_total
            overall_ratio = total_class_0 / total_samples
            
            # Score based on:
            # 1. How close ratios are to overall ratio
            # 2. Whether minimum requirements are met
            # 3. How close to target split sizes (10% each)
            ratio_score = abs(val_ratio - overall_ratio) + abs(test_ratio - overall_ratio)
            min_req_score = 0
            if val_class_0 < min_class_0_needed or val_class_1 < min_class_1_needed:
                min_req_score += 100
            if test_class_0 < min_class_0_needed or test_class_1 < min_class_1_needed:
                min_req_score += 100
                
            size_score = abs(val_total - 0.1 * total_samples) + abs(test_total - 0.1 * total_samples)
            
            total_score = ratio_score + min_req_score + size_score * 0.1
            
            if total_score < best_score:
                best_score = total_score
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
                  target: str,
                  class_mapping: dict = None) -> Tuple[list, list, list]:
    """
    Split the dataset into train, val, and test sets while 
    
    Parameters
    ----------
    input_folder: str
        Path to the folder containing input images.
    annotation_folder: str
        Path to the folder containing annotation files.
    target: str
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
    output_file = output_dir / f"split_distribution_{target}_{timestamp}.txt"
    
    split_info = []
    split_info.append(f"Dataset Split Information - {timestamp}\n")
    split_info.append(f"Found {len(all_input_images)} {target} images in {input_folder}\n")
    
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
           
def split_yolo_data(annotation_folder: Path, target: str):
    """
    This function prepares the dataset for YOLO training by splitting the images into train, val, and test sets.
    
    Parameters:
    ----------
    annotation_folder: Path
        Path to the directory containing label files.
    target: str
        The target object for YOLO detection or classification.
    """
    logging.info(f"Starting dataset preparation for {target}")

    try:
        # Define mappings for different target types
        target_mappings = {
            "face_cls": {
                1: ("adult_face", DetectionPaths.face_images_input_dir),
                0: ("child_face", DetectionPaths.face_images_input_dir)
            },
            "person_cls": {
                1: ("adult_person", DetectionPaths.person_images_input_dir),
                0: ("child_person", DetectionPaths.person_images_input_dir)
            },
            "gaze_cls": {
                0: ("no_gaze", DetectionPaths.gaze_images_input_dir),
                1: ("gaze", DetectionPaths.gaze_images_input_dir)
            }
        }       
        # Get annotated frames
        total_images = get_total_number_of_annotated_frames(annotation_folder)
        
        if target in target_mappings:
            # Get source directories based on target type
            input_folder = target_mappings[target][0][1]  # Use first mapping's input dir
            class_mapping = target_mappings[target]
            # Get custom splits
            train_images, val_images, test_images = split_dataset(
                input_folder, annotation_folder, target, class_mapping)
                
            # Process each split
            for split_name, split_images in [("train", train_images), 
                                           ("val", val_images), 
                                           ("test", test_images)]:
                # Separate images into classes
                for class_id in [0, 1]:
                    class_images = [img for img in split_images 
                                  if get_class(img, annotation_folder) == class_id]
                    if class_images:
                        target = target_mappings[target][class_id][0]
                        successful, failed = move_images(
                            target=target,
                            image_names=class_images,
                            split_type=split_name,
                            label_path=annotation_folder,
                            n_workers=4
                        )   
                        logging.info(f"{split_name} {target}: Moved {successful}, Failed {failed}")
                    else:
                        target = target_mappings[target][class_id][0]
                        logging.warning(f"No {target} images for {split_name}")
        else:         
            # Multi-class detection case
            df = get_class_distribution(total_images, annotation_folder, target)
            # split data grouped by id
            train, val, test, *_ = multilabel_stratified_split(df, target=target)

            # Move images for each split
            for split_name, split_set in [("train", train), 
                                        ("val", val), 
                                        ("test", test)]:
                if split_set:
                    successful, failed = move_images(
                        target=target,
                        image_names=split_set,
                        split_type=split_name,
                        label_path=annotation_folder,
                        n_workers=4
                    )
                    logging.info(f"{split_name}: Moved {successful}, Failed {failed}")
                else:
                    logging.warning(f"No images for {split_name} split")
    
    except Exception as e:
        logging.error(f"Error processing target {target}: {str(e)}")
        raise
    
    logging.info(f"\nCompleted dataset preparation for {target}")
    
def main(target: str):
    """
    Main function to prepare the dataset for model training.
    
    Parameters:
    ----------
    target : str
        The target type for YOLO (e.g., "person" or "face").
    """
    path_mapping = {
        "person_face": DetectionPaths.person_face_labels_input_dir,
        "all": DetectionPaths.all_labels_input_dir,
        "person_cls": ClassificationPaths.person_labels_input_dir,
        "face_cls": ClassificationPaths.face_labels_input_dir,
        "gaze_cls": ClassificationPaths.gaze_labels_input_dir
    }
    label_path = path_mapping[target]
    split_yolo_data(label_path, target)
    logging.info("Dataset preparation for YOLO completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for model training.")
    parser.add_argument("--target", choices=VALID_TARGETS, required=True, help="Specify the YOLO target type")
    
    args = parser.parse_args()
    main(args.target)
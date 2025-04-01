import shutil
import random
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from constants import DetectionPaths, YoloPaths, ResNetPaths, BasePaths
from config import TrainingConfig
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Optional, Tuple

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

def get_binary_class_distribution(total_images: list, annotation_folder: Path, target: str) -> (set, set):
    """
    This function reads label files and iterates over the detections and assign them to the correct class.
      
    Parameters:
    ----------
    total_images: list
        List of image names.
    annotation_folder: Path
        Path to the directory containing label files.
    target: str
        The target type for binary classification (e.g., "gaze" or "person" or "face").
    
    Returns:
    -------
    images_class_0: set
        Set of image names with class 0.
    images_class_1: set
        Set of image names with class 1.    
    """
    images_class_0= set()
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
                    if class_id == "0": 
                        images_class_0.add(f"{image_file.stem}_{class_description}_{i}.jpg")
                    elif class_id == "1":  
                        images_class_1.add(f"{image_file.stem}_{class_description}_{i}.jpg")
    
    total_gaze = len(images_class_1) + len(images_class_0)
    # log class distribution
    logging.info(f"Class 0: {len(images_class_0)} images")
    logging.info(f"Class 1: {len(images_class_1)} images")
    logging.info(f"Total: {total_gaze} images")
    
    return images_class_0, images_class_1

def binary_stratified_split(image_sets: list, yolo_target: str, train_ratio: float = TrainingConfig.train_test_split_ratio):
    """
    Performs stratified split for binary classification while maintaining class distributions.
    Only balances training set while keeping original distribution for val/test.
    
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
        "class_0": (list of train, list of val, list of test),
        "class_1": (list of train, list of val, list of test)
    }
    """
    if len(image_sets) != 2:
        raise ValueError("Image_sets must contain exactly two sets")
        
    # Get class labels
    if yolo_target == "gaze":
        class_labels = YoloPaths.gaze_classes
    elif yolo_target == "person":
        class_labels = ResNetPaths.person_classes
    elif yolo_target == "face":
        class_labels = ResNetPaths.face_classes
    else:
        raise ValueError(f"Invalid yolo_target: {yolo_target}")

    result = {}
    val_ratio = (1 - train_ratio) / 2

    # Get sizes of both classes
    class_0_size = len(image_sets[0])
    class_1_size = len(image_sets[1])
    
    # Log initial distribution
    logging.info(f"\nInitial class distribution:")
    logging.info(f"Class 0: {class_0_size} images")
    logging.info(f"Class 1: {class_1_size} images")

    # Process each class separately
    for idx, (label, image_set) in enumerate(zip(class_labels, image_sets)):
        images = list(image_set)
        if not images:
            logging.warning(f"No images found for class {label}")
            result[label] = ([], [], [])
            continue

        # Calculate split sizes
        train_size = int(len(images) * train_ratio)
        val_size = int(len(images) * val_ratio)
        
        # Shuffle with fixed seed for reproducibility
        random.seed(TrainingConfig.random_seed)
        random.shuffle(images)
        
        # Split into train/val/test
        train = images[:train_size]
        val = images[train_size:train_size + val_size]
        test = images[train_size + val_size:]

        result[label] = (train, val, test)
        
        # Log distributions
        logging.info(f"\nClass '{label}' split distribution:")
        logging.info(f"Train: {len(train)} ({len(train)/len(images):.2%})")
        logging.info(f"Val: {len(val)} ({len(val)/len(images):.2%})")
        logging.info(f"Test: {len(test)} ({len(test)/len(images):.2%})")

    # Balance training sets by downsampling majority class
    train_0, val_0, test_0 = result[class_labels[0]]
    train_1, val_1, test_1 = result[class_labels[1]]
    
    min_train_size = min(len(train_0), len(train_1))
    
    # Downsample majority class in training set
    if len(train_0) > min_train_size:
        random.seed(TrainingConfig.random_seed)
        train_0 = random.sample(train_0, min_train_size)
        result[class_labels[0]] = (train_0, val_0, test_0)
    elif len(train_1) > min_train_size:
        random.seed(TrainingConfig.random_seed)
        train_1 = random.sample(train_1, min_train_size)
        result[class_labels[1]] = (train_1, val_1, test_1)
    
    # Log final balanced training distribution
    logging.info(f"\nFinal balanced training distribution:")
    logging.info(f"Class 0 training samples: {len(train_0)}")
    logging.info(f"Class 1 training samples: {len(train_1)}")

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
    logging.info(f"Starting dataset preparation for {yolo_target}")
    total_images = get_total_number_of_annotated_frames(label_path)

    try:
        if yolo_target in ["person", "face", "gaze"]:
            # Handle binary classification cases
            class_mapping = {
                "person": ["child_person", "adult_person"],
                "face": ["child_face", "adult_face"],
                "gaze": ["no_gaze", "gaze"]
            }
            images_class_0, images_class_1 = get_binary_class_distribution(
                total_images, label_path, yolo_target
            )
            
            splits_dict = binary_stratified_split(
                (images_class_0, images_class_1), yolo_target
            )
            
            # Process each binary class
            for binary_class in class_mapping[yolo_target]:
                train, val, test = splits_dict[binary_class]
                logging.info(f"\nProcessing {binary_class}:")
                
                # Move images for each split using parallel processing
                for split_name, split_set in [("train", train), ("val", val), ("test", test)]:
                    if split_set:  # Only process if we have images
                        successful, failed = move_images(
                            yolo_target=binary_class,
                            image_names=split_set,
                            split_type=split_name,
                            label_path=label_path,
                            n_workers=4  # Adjust based on your system
                        )
                        logging.info(f"{split_name}: Moved {successful} images, Failed {failed}")
                    else:
                        logging.warning(f"No images for {binary_class} {split_name} split")
                        
        else:
            # Handle multi-class cases
            df = get_class_distribution(total_images, label_path, yolo_target)
            
            train, val, test, train_df, val_df, test_df = multilabel_stratified_split(
                df, yolo_target=yolo_target
            )
            
            # Log distributions before moving
            log_all_split_distributions(train_df, val_df, test_df, yolo_target)
            
            # Move images for each split
            total_successful = total_failed = 0
            for split_name, split_set in [("train", train), ("val", val), ("test", test)]:
                if split_set:  # Only process if we have images
                    successful, failed = move_images(
                        yolo_target=yolo_target,
                        image_names=split_set,
                        split_type=split_name,
                        label_path=label_path,
                        n_workers=4  # Adjust based on your system
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
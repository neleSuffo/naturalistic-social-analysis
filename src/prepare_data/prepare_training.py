import shutil
import random
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from constants import DetectionPaths, YoloPaths, ResNetPaths
from config import TrainingConfig
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

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
            9: "food", 10: "other_object"
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
            10: "interacted_animal", 11: "interacted_food", 12: "animal", 13: "food"
        },
        "person_face_object": {
            0: "person", 1: "face", 2: "child_body_parts", 3: "book",
            4: "toy", 5: "kitchenware", 6: "screen", 7: "other_object",
            8: "food", 9: "animal"
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
    This function splits the images into train, val, and test sets based on the class distribution. 
    It also balances the train set.
    
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
    val_ratio = (1 - train_ratio) / 2
    
    # Split each set separately
    if len(image_sets) != 2:
        raise ValueError("Image_sets must contain exactly two sets.")
    result = {}
    if yolo_target == "gaze":
        class_labels = YoloPaths.gaze_classes
    elif yolo_target == "person":
        class_labels = ResNetPaths.person_classes
    elif yolo_target == "face":
        class_labels = ResNetPaths.face_classes
    
    # Convert tuple elements to lists and undersample one class if necessary
    
    num_class_0_images = len(list(image_sets[0]))
    num_class_1_images = len(list(image_sets[1]))
    
    # Undersample the larger class
    if num_class_0_images > num_class_1_images:
        logging.info(f"Undersampling class 0 to match class 1")
        class_0_images_new = random.sample(list(image_sets[0]), num_class_1_images)
        class_1_images_new = list(image_sets[1])
    elif num_class_1_images > num_class_0_images:
        logging.info(f"Undersampling class 1 to match class 0")
        class_1_images_new = random.sample(list(image_sets[1]), num_class_0_images)
        class_0_images_new = list(image_sets[0])
    else:
        logging.info(f"Class 0 and class 1 have the same number of images")
        class_0_images_new = list(image_sets[0])
        class_1_images_new = list(image_sets[1])
    
    # Create new image sets
    new_image_sets = [class_0_images_new, class_1_images_new]

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
    
def stratified_split(df: pd.DataFrame, train_ratio=TrainingConfig.train_test_split_ratio, random_seed=TrainingConfig.random_seed, yolo_target=None):
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
    if yolo_target in ["child_face", "adult_face", "adult_person", "child_person"]:
        paths = ResNetPaths.get_target_paths(yolo_target, split_type)
    else:
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
                    
            elif yolo_target in ["gaze", "no_gaze", "child_face", "adult_face", "adult_person", "child_person"]:
                # Get correct input directory based on target
                if yolo_target in ["gaze", "no_gaze"]:
                    input_dir = DetectionPaths.gaze_images_input_dir
                elif yolo_target in ["child_face", "adult_face"]:
                    input_dir = DetectionPaths.face_images_input_dir
                elif yolo_target in ["adult_person", "child_person"]:
                    input_dir = DetectionPaths.person_images_input_dir
                    
                image_src = input_dir / image_name
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
            for binary_class in class_mapping[yolo_target]:
                train, val, test = splits_dict[binary_class]
                for split_name, split_set in (("train", train), ("val", val), ("test", test)):
                    move_images(binary_class, split_set, split_name, label_path)
        else:
            # Handle multi-class cases
            df = get_class_distribution(total_images, label_path, yolo_target)
            train, val, test, train_df, val_df, test_df = stratified_split_all(
                df, yolo_target=yolo_target
            )
            log_all_split_distributions(train_df, val_df, test_df, yolo_target)
            for split_name, split_set in (("train", train), ("val", val), ("test", test)):
                move_images(yolo_target, split_set, split_name, label_path)
    except Exception as e:
        logging.error(f"Error processing target {yolo_target}: {str(e)}")
        raise
            
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
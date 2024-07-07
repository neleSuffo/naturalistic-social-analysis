import shutil
import random
import logging
from pathlib import Path
from src.projects.social_interactions.common.constants import DetectionPaths, TrainParameters, YoloParameters as Yolo, MtcnnParameters as Mtcnn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def check_annotations(
    jpg_dir: Path, 
    txt_dir: Path
) -> None:
    """
    This function checks if for every .jpg file there is a .txt file with the same name. 
    If not, it creates an empty .txt file.

    Parameters
    ----------
    jpg_dir : Path
        the directory with the .jpg files
    txt_dir : Path
        the directory with the .txt files
    """
    # Get the list of all .jpg and .txt files
    jpg_files = {f.stem for f in jpg_dir.glob('*.jpg')}
    txt_files = {f.stem for f in txt_dir.glob('*.txt')}
    
    # Find the missing .txt files
    missing_txt_files = jpg_files - txt_files
    
    # Create empty .txt files for the missing .jpg files
    for file in missing_txt_files:
        (txt_dir / f"{file}.txt").touch()


def split_dataset(
    file_dir_jpg: Path,
    split_ratio: float
    ) -> tuple:
    """
    This function splits the dataset into training and testing sets
    and returns the list of training and testing files.

    Parameters
    ----------
    file_dir_jpg : str
        the directory with the .jpg files
    split_ratio : float
        the ratio to split the dataset
    
    Returns
    -------
    tuple
        the list of training and testing files
    """
    logging.info(f"Splitting dataset with ratio {split_ratio}")
    # Get all image files and their corresponding annotation files
    files = list(file_dir_jpg.glob('*.jpg'))

    # Set a random seed for reproducibility
    random.seed(TrainParameters.random_seed)
    random.shuffle(files)

    # Calculate the split index
    split_index = int(len(files) * split_ratio)
    
    # Split the files into training and testing sets
    train_files = files[:split_index]
    val_files = files[split_index:]
    
    logging.info(f"Dataset split completed. Training files: {len(train_files)}, Validation files: {len(val_files)}")
    return train_files, val_files


def move_yolo_files(
    files_to_move_lst: list, 
    dest_dir_images: Path,
    dest_dir_labels: Path,
    ) -> None:
    """
    This function moves files from the source directory to the destination directory
    and deletes the empty source directory.

    Parameters
    ----------
    files_to_move_lst : list
        the list of files to move
    dest_dir_images : Path
        the destination directory for the image files
    dest_dir_labels : Path
        the destination directory for the label files
    """        
    logging.info(f"Moving YOLO files to {dest_dir_images} and {dest_dir_labels}")
    # Move the files to the new directory
    for file_path in files_to_move_lst:
        # Construct the full source paths for the image and label
        src_label_path = Yolo.labels_input / file_path.with_suffix('.txt').name

        # Construct the destination paths for the image and label
        dest_image_path = dest_dir_images / file_path.name
        dest_label_path = dest_dir_labels / file_path.with_suffix('.txt').name

        # Copy the images and move the label to their new destinations
        shutil.copy(file_path, dest_image_path)
        src_label_path.rename(dest_label_path)
    logging.info("YOLO files moved successfully")


def prepare_yolo_dataset(
    destination_dir: Path,
    train_files: list,
    val_files: list,
):
    """
    This function moves the training and testing files to the new yolo directories.

    Parameters
    ----------
    destination_dir : Path
        the destination directory to store the training and testing sets
    train_files : list
        the list of training files
    val_files : list
        the list of testing files
    """
    # Define source directory and new directories for training
    train_dir_images = destination_dir / "images/train"
    val_dir_images = destination_dir / "images/val"
    train_dir_labels = destination_dir / "labels/train"
    val_dir_labels = destination_dir / "labels/val"
    
    # Create necessary directories if they don't exist
    for path in [train_dir_images, val_dir_images, train_dir_labels, val_dir_labels]:
        path.mkdir(parents=True, exist_ok=True)
    
    # Move the files to the new directories
    move_yolo_files(train_files, train_dir_images, train_dir_labels)
    move_yolo_files(val_files, val_dir_images, val_dir_labels)  


def move_mtcnn_files(
    train_files: list,
    val_files: list, 
    train_dir_images: Path,
    val_dir_images: Path,
    train_labels_path: Path,
    val_labels_path: Path,
    )-> None:
    """
    This function moves the training and testing files to the new mtcnn directories.

    Parameters
    ----------
    train_files : list
        the list of training files
    val_files : list
        the list of testing files
    train_dir_images: Path
        the directory to store the training images
    val_dir_images: Path
        the directory to store the testing images
    train_labels_path: Path
        the path to store the training labels
    val_labels_path: Path
        the path to store the testing labels
    """
    logging.info(f"Moving MTCNN files to {train_dir_images}, {val_dir_images}, {train_labels_path} and {val_labels_path}")
    # Move the images to the training and testing directories
    for file_path in train_files:
        src_image_path = DetectionPaths.images_input / file_path.name
        dest_image_path = train_dir_images / file_path.name
        src_image_path.rename(dest_image_path)
    for file_path in val_files:
        src_image_path = DetectionPaths.images_input / file_path.name
        dest_image_path = val_dir_images / file_path.name
        src_image_path.rename(dest_image_path)
    
    # Convert lists to sets and extract the file names
    train_set = set(train_files)
    val_set = set(val_files)
    file_names_in_train_set = {path.name.rsplit('.', 1)[0] for path in train_set}
    file_names_in_val_set = {path.name.rsplit('.', 1)[0] for path in val_set}

    # Move the labels to the training and testing directories
    with Mtcnn.labels_input.open('r') as original_file, train_labels_path.open('w') as train_file, val_labels_path.open('w') as validation_file:
        for line in original_file:
            image_file_name = line.split()[0]  # Assuming the file name is the first element
            if image_file_name in file_names_in_train_set:
                train_file.write(line)
            elif image_file_name in file_names_in_val_set:
                validation_file.write(line)
    logging.info("MTCNN files moved successfully")


def prepare_mtcnn_dataset(
    destination_dir: Path,
    train_files: list,
    val_files: list,
):
    """
    This function moves the training and testing files to the new yolo directories.

    Parameters
    ----------
    destination_dir : Path
        the destination directory to store the training and testing sets
    train_files : list
        the list of training files
    val_files : list
        the list of testing files
    """
    # Define source directory and new directories for training
    train_dir_images = destination_dir / "train/images"
    train_labels_path = destination_dir / "train/train.txt"
    val_dir_images = destination_dir / "val/images"
    val_labels_path = destination_dir / "val/val.txt"

    # Create necessary directories if they don't exist
    for path in [train_dir_images, val_dir_images, train_labels_path.parent, val_labels_path.parent]:
        path.mkdir(parents=True, exist_ok=True)
    
    # Move the files to the new directories
    move_mtcnn_files(train_files, 
                    val_files, 
                    train_dir_images, 
                    val_dir_images,
                    train_labels_path, 
                    val_labels_path,
    )
    
    # Remove the empty directories and the mtcnn labels file
    shutil.rmtree(DetectionPaths.images_input)
    shutil.rmtree(Yolo.labels_input)
    Mtcnn.labels_input.unlink(missing_ok=True)

def main():
    check_annotations(DetectionPaths.images_input, Yolo.labels_input)
    # Move label files and delete empty labels directory
    train_files, val_files = split_dataset(DetectionPaths.images_input, TrainParameters.train_test_split)
    prepare_yolo_dataset(Yolo.data_input, train_files, val_files)
    prepare_mtcnn_dataset(Mtcnn.data_input, train_files, val_files)


if __name__ == "__main__":
    main()
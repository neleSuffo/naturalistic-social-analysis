import os
import shutil
import random
from projects.social_interactions.src.common.constants import DetectionPaths, TrainParameters


def check_annotations_and_videos(
    jpg_dir: str, 
    txt_dir: str
) -> None:
    """This

    Parameters
    ----------
    jpg_dir : str
        the directory with .jpg files
    txt_dir : str
        the directory with .txt files
    """
    # Get the list of all .jpg and .txt files
    jpg_files = [f for f in os.listdir(jpg_dir) if f.endswith('.jpg')]
    txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]

    # Remove the file extensions
    jpg_files_no_ext = [os.path.splitext(f)[0] for f in jpg_files]
    txt_files_no_ext = [os.path.splitext(f)[0] for f in txt_files]
    
    # List to store .jpg files that are missing a .txt file
    missing_txt_files = []  

    # Check if for every .jpg file there is a .txt file with the same name
    for jpg_file in jpg_files_no_ext:
        if jpg_file not in txt_files_no_ext:
            missing_txt_files.append(jpg_file)
    
    # Create empty .txt files for the missing .jpg files
    for jpg_file in missing_txt_files:
        open(os.path.join(txt_dir, jpg_file + '.txt'), 'a').close()


def split_dataset(
    file_dir_jpg: str,
    destination_dir: str,
    split_ratio: float
    ) -> None:
    """
    This function splits the dataset into training and testing sets.

    Parameters
    ----------
    file_dir_jpg : str
        the directory containing the image files
    file_dir_txt : str
        the directory containing the annotation files
    destination_dir : str
        the destination directory to store the training and testing sets
    split_ratio : float
        the ratio to split the dataset
    """
    # Define source directory and new directories for training
    train_dir = os.path.join(destination_dir, "train")
    val_dir = os.path.join(destination_dir, "val")

    # Create necessary directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Get all image files and their corresponding annotation files
    files = [f for f in os.listdir(file_dir_jpg) if f.endswith('.jpg')]

    # Shuffle the files randomly
    random.shuffle(files)

    # Calculate the split index
    split_index = int(len(files) * split_ratio)
    
    # Split the files into training and testing sets
    train_files_jpg = files[:split_index]
    val_files_jpg = files[split_index:]
    
    train_files_txt = [file_name.replace('.jpg', '.txt') for file_name in train_files_jpg]
    val_files_txt = [file_name.replace('.jpg', '.txt') for file_name in val_files_jpg]


    def move_files(
        src_dir: str,
        files_to_move_lst: list, 
        dest_dir: str
    ) -> None:
        """
        This function moves files from the source directory to the destination directory and deletes the empty source directory.

        Parameters
        ----------
        src_dir : str
            the source directory
        files_to_move_lst : list
            the list of files to move
        dest_dir : str
            the destination directory
        """
        # Create the new directory if it doesn't exist
        os.makedirs(dest_dir, exist_ok=True)
        
        # Move the files to the new directory
        for file_name in files_to_move_lst:
            source_file = os.path.join(src_dir, file_name)
            destination_file = os.path.join(dest_dir, file_name)
            
            # Move the file to the new directory
            shutil.move(source_file, destination_file)
        
        # Remove the directory if it is empty
        if not os.listdir(src_dir):
            os.rmdir(src_dir)

    # Move the files to the new directories
    move_files(DetectionPaths.images_input,train_files_jpg, train_dir)
    move_files(DetectionPaths.images_input, val_files_jpg, val_dir)
    move_files(DetectionPaths.labels_input, train_files_txt, train_dir)
    move_files(DetectionPaths.labels_input, val_files_txt, val_dir)   


def main():
    check_annotations_and_videos(DetectionPaths.images_input, DetectionPaths.labels_input)
    # Move label files and delete empty labels directory
    split_dataset(DetectionPaths.images_input, DetectionPaths.yolo_input, TrainParameters.train_test_split)


if __name__ == "__main__":
    main()
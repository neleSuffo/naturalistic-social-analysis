import random
import shutil
from pathlib import Path

def balance_dataset(annotation_dir: Path = YoloPaths.face_data_input_dir / "labels/train",
                    images_dir: Path = YoloPaths.face_data_input_dir / "images/train",
                    balanced_images_dir: Path = YoloPaths.face_balanced_images_input_dir,
                    balanced_annotations_dir: Path = YoloPaths.face_balanced_labels_input_dir):
    """
    This function balances a dataset by randomly selecting an equal number of images with and without faces.
    
    Parameters:
    ----------
    annotation_dir: Path
        The directory containing the annotation files.
    images_dir: Path
        The directory containing the image files.
    balanced_images_dir: Path
        The directory to save the balanced image files.
    balanced_annotations_dir: Path
        The directory to save the balanced annotation files.
    """
    # Lists to hold image filenames
    images_with_faces = []
    images_without_faces = []

    # Iterate over annotation files to classify images
    for annotation_path in annotations_dir.iterdir():
        if annotation_path.suffix == '.txt':
            image_path = images_dir / (annotation_path.stem + '.jpg')
            # Check if the annotation file is not empty
            if annotation_path.stat().st_size > 0:
                images_with_faces.append((image_path, annotation_path))
            else:
                images_without_faces.append((image_path, annotation_path))

    # Determine the number of images with faces
    num_faces = len(images_with_faces)
    logging.info(f"Found {num_faces} images with faces.")

    # Randomly select an equal number of images without faces
    images_without_faces_sample = random.sample(images_without_faces, num_faces)

    # Combine the lists to form the balanced dataset
    balanced_dataset = images_with_faces + images_without_faces_sample

    # Ensure the balanced dataset directories exist
    balanced_images_dir.mkdir(parents=True, exist_ok=True)
    balanced_annotations_dir.mkdir(parents=True, exist_ok=True)

    # Copy the selected files to the balanced dataset directories
    for image_path, annotation_path in balanced_dataset:
        shutil.copy(image_path, balanced_images_dir / image_path.name)
        shutil.copy(annotation_path, balanced_annotations_dir / annotation_path.name)

    logging.info(f"Balanced dataset created with {num_faces} images with faces and {num_faces} images without faces.")
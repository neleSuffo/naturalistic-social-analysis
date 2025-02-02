import json
import os
import logging
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

def extract_video_names(annotation_dir):
    """
    Extract unique video identifiers from annotation files.

    Args:
        annotation_dir (Path): Directory containing annotation .txt files.

    Returns:
        set: A set of unique video identifiers.
    """
    video_names = set()
    for annotation_file in annotation_dir.glob('*.txt'):
        # Extract the video identifier by removing the last part of the filename
        video_id = '_'.join(annotation_file.stem.split('_')[:-1])
        video_names.add(video_id)
    return video_names

def process_images(root_dir, excluded_videos, model):
    """
    Processes images in the specified root directory, applying YOLO11 to detect faces and compiling statistics.

    Args:
        root_dir (str or Path): Root directory containing subdirectories of images.
        excluded_videos (set): Set of video names to be excluded from processing.
        model (YOLO): YOLO11 model instance.

    Returns:
        dict: A dictionary containing statistics on face detections.
    """
    face_counts = {}
    total_images = 0
    num_videos = 0

    # Iterate through each subdirectory (video folder) in the root directory
    logging.info(f"Found {len(list(root_dir.iterdir()))} video folders")
    for video_folder in root_dir.iterdir():
        logging.info(f"Processing {video_folder.name}")
        if video_folder.is_dir() and video_folder.name not in excluded_videos:
            # Get a list of all image files in the current video folder
            image_files = list(video_folder.glob("*.jpg"))  # Adjust the pattern if your images have a different extension

            # Process each image file
            for image_file in tqdm(image_files, desc=f"Processing {video_folder.name}", unit="image"):
                total_images += 1
                results = model(image_file)
                num_faces = len(results[0].boxes)
                face_counts[num_faces] = face_counts.get(num_faces, 0) + 1
            num_videos += 1

    # Calculate percentages
    face_distribution = {count: {"num_images": num, "percentage": (num / total_images) * 100}
                         for count, num in face_counts.items()}

    # Compile summary statistics
    summary = {
        "total_images": total_images,
        "num_videos": num_videos,
        "face_distribution": face_distribution,
    }

    return summary

def save_statistics(statistics, output_json, output_txt):
    """
    Save statistics to JSON and text files.

    Args:
        statistics (dict): Dictionary containing statistics.
        output_json (Path): Path to the output JSON file.
        output_txt (Path): Path to the output text file.
    """
    # Save to JSON
    with output_json.open('w') as json_file:
        json.dump(statistics, json_file, indent=4)

    # Save to text file
    with output_txt.open('w') as txt_file:
        txt_file.write(f"Total number of images: {statistics['total_images']}\n")
        for count, num_images in statistics['face_counts'].items():
            percentage = statistics['face_counts_percentage'][count]
            txt_file.write(f"Images with {count} faces: {num_images} ({percentage:.2f}%)\n")

if __name__ == "__main__":
    # Set thread limits
    os.environ['OMP_NUM_THREADS'] = '4'  # OpenMP threads
    torch.set_num_threads(4)  # PyTorch threads
    # Define paths
    annotation_directory = Path('/home/nele_pauline_suffo/ProcessedData/yolo_face_labels')
    image_directory = Path('/home/nele_pauline_suffo/ProcessedData/quantex_videos_processed')
    output_json_path = Path('/home/nele_pauline_suffo/outputs/yolo_face_detections/quantex_full_video_run/statistics.json')
    output_txt_path = Path('/home/nele_pauline_suffo/outputs/yolo_face_detections/quantex_full_video_run/statistics.txt')

    # Load YOLO11 model
    yolo_model = YOLO('/home/nele_pauline_suffo/models/yolo11_face_detection.pt')

    # Extract video names to exclude
    excluded_video_names = extract_video_names(annotation_directory)

    # Process images and get statistics
    stats = process_images(image_directory, excluded_video_names, yolo_model)

    # Save statistics to files
    save_statistics(stats, output_json_path, output_txt_path)
import json
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

def process_images(image_dir, excluded_videos, model):
    """
    Process images to detect faces and compile statistics.

    Args:
        image_dir (Path): Directory containing images.
        excluded_videos (set): Set of video identifiers to exclude from processing.
        model (YOLO): Preloaded YOLO model for face detection.

    Returns:
        dict: A dictionary with counts of detected faces.
    """
    face_counts = defaultdict(int)
    total_images = 0

    # Iterate through all image files in the directory and subdirectories
    image_files = list(image_dir.rglob('*.jpg')) + list(image_dir.rglob('*.png'))
    for image_path in tqdm(image_files, desc="Processing images"):
        # Check if the image belongs to an excluded video
        if any(video_id in image_path.stem for video_id in excluded_videos):
            continue

        # Load and process the image with YOLO
        results = model(image_path)
        num_faces = len(results[0].boxes)
        face_counts[num_faces] += 1
        total_images += 1

    # Calculate percentages
    face_counts_percentage = {k: (v / total_images) * 100 for k, v in face_counts.items()}

    return {
        'total_images': total_images,
        'face_counts': dict(face_counts),
        'face_counts_percentage': face_counts_percentage
    }

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
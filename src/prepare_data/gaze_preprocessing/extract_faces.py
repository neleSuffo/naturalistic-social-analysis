import os
import cv2
import logging
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from constants import MtcnnPaths, DetectionPaths

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the minimum level of messages to display
    format="%(asctime)s - %(levelname)s - %(message)s",  # Define the log message format
    datefmt="%Y-%m-%d %H:%M:%S",  # Define the date format
)

def crop_faces_from_labels(
    labels_file: str = MtcnnPaths.labels_file_path,
    rawframe_dir: str = DetectionPaths.images_input_dir,
    output_dir: str = MtcnnPaths.faces_dir,
    labels_output_file: str = MtcnnPaths.face_labels_file_path,
    progress_file: str = MtcnnPaths.progress_file_path,  # File to track progress
    missing_frames_file: str = MtcnnPaths.missing_frames_file_path,  # File to track missing frames
):
    """
    Crop faces from raw frames using bounding box information in the labels file and save them.

    Args:
        labels_file (str): Path to the labels file.
        rawframe_dir (str): Directory containing raw frames.
        output_dir (str): Directory to save cropped faces and labels.
        labels_output_file (str): File to save cropped face labels.
        progress_file (str): File to track processed images.
        missing_frames_file (str): File to log missing raw frames.
    """
    cv2.setNumThreads(1)
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load progress
    processed_images = set()
    if Path(progress_file).exists():
        with open(progress_file, 'r') as progress_reader:
            processed_images.update(line.strip() for line in progress_reader)

    annotations_by_image = defaultdict(list)

    with open(labels_file, 'r') as f, open(missing_frames_file, 'a') as missing_writer:
        for line in tqdm(f.readlines(), desc="Processing Labels"):
            parts = line.strip().split()
            image_name = parts[0]

            # Skip already processed images
            if image_name in processed_images:
                continue

            bbox_and_gaze = parts[1:]

            frame_path = Path(rawframe_dir) / image_name
            if not frame_path.exists():
                logging.warning(f"Frame {frame_path} not found. Logging to missing frames.")
                missing_writer.write(f"{image_name}\n")
                continue

            # Read the frame
            frame = cv2.imread(str(frame_path))
            if frame is None:
                logging.error(f"Failed to load {frame_path}. Logging to missing frames.")
                missing_writer.write(f"{image_name}\n")
                continue

            for item in bbox_and_gaze:
                values = item.split(',')
                bbox_values = list(map(float, values[:4]))  # First 4 values are bbox
                gaze_label = int(values[4])  # The 5th value is the gaze label

                x1, y1, w, h = bbox_values
                x2, y2 = x1 + w, y1 + h

                # Crop the face
                cropped_face = frame[int(y1):int(y2), int(x1):int(x2)]
                if cropped_face.size == 0:
                    logging.warning(f"Empty crop for {frame_path}. Skipping.")
                    continue

                annotations_by_image[image_name].append((cropped_face, gaze_label))

            # Log processed image
            with open(progress_file, 'a') as progress_writer:
                progress_writer.write(f"{image_name}\n")

    with open(labels_output_file, 'a') as label_writer:
        for image_name, faces in annotations_by_image.items():
            for idx, (cropped_face, gaze_label) in enumerate(faces):
                # Save the cropped face
                cropped_face_name = f"{image_name.split('.')[0]}_face_{idx}.jpg"
                cropped_face_path = output_dir / cropped_face_name
                cv2.imwrite(str(cropped_face_path), cropped_face)

                # Write label information
                label_writer.write(f"{cropped_face_path} {gaze_label}\n")

    logging.info(f"Cropped faces saved to {output_dir}")
    logging.info(f"Labels saved to {labels_output_file}")
    logging.info(f"Progress tracked in {progress_file}")
    logging.info(f"Missing frames logged in {missing_frames_file}")


crop_faces_from_labels()
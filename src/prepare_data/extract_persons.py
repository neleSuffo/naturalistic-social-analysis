import os
import cv2
import logging
from pathlib import Path
from tqdm import tqdm
from constants import DetectionPaths, YoloPaths

def crop_persons_from_labels(
    labels_input_dir: Path,
    rawframe_dir: Path,
    output_dir: Path,
    progress_file: Path,
    missing_frames_file: Path
):
    """ 
    This function reads YOLO annotations and crops persons from rawframes.
    
    Parameters
    ----------
    labels_input_dir : Path
        Directory containing YOLO annotations in txt format
    rawframe_dir : Path
        Directory containing rawframes
    output_dir : Path
        Directory to save cropped persons
    progress_file : Path
        File to track progress
    missing_frames_file : Path
        File to log missing frames
    """
    cv2.setNumThreads(1)
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_dir = progress_file.parent
    progress_dir.mkdir(parents=True, exist_ok=True)

    # Load or initialize progress tracking
    processed_images = set()
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            processed_images = set(line.strip() for line in f)
        logging.info(f"Loaded {len(processed_images)} processed images")

    # Get all txt files in labels directory
    annotation_files = list(labels_input_dir.glob('*.txt'))

    for ann_file in tqdm(annotation_files, desc="Processing annotations"):
        image_name = ann_file.stem + '.jpg'
        
        # Skip if already processed
        if image_name in processed_images:
            continue

        # Construct image path
        video_folder = '_'.join(image_name.split('_')[:8])
        image_path = rawframe_dir / video_folder / image_name

        if not image_path.exists():
            logging.warning(f"Image {image_path} not found")
            with open(missing_frames_file, 'a') as f:
                f.write(f"{image_name}\n")
            continue

        # Read the image
        frame = cv2.imread(str(image_path))
        if frame is None:
            logging.error(f"Failed to load {image_path}")
            with open(missing_frames_file, 'a') as f:
                f.write(f"{image_name}\n")
            continue

        frame_height, frame_width = frame.shape[:2]

        # Read annotations
        with open(ann_file, 'r') as f:
            lines = f.readlines()
            # Process each line in the annotation file
            for idx, line in enumerate(lines):
                try:
                    # Parse YOLO format: class x_center y_center width height
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    
                    # Only process person classes (0 = person, 1 = reflection)
                    if class_id not in [0, 1]:
                        continue
                    
                    # Convert YOLO coordinates to pixel coordinates
                    x1 = int((x_center - width/2) * frame_width)
                    y1 = int((y_center - height/2) * frame_height)
                    w = int(width * frame_width)
                    h = int(height * frame_height)
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    w = min(w, frame_width - x1)
                    h = min(h, frame_height - y1)

                    # Crop the person
                    cropped_person = frame[y1:y1+h, x1:x1+w]
                    if cropped_person.size == 0:
                        logging.warning(f"Empty crop for {image_path}")
                        continue

                    # Save cropped person with same name as original
                    person_output_path = output_dir / f"{ann_file.stem}_person_{idx}.jpg"
                    cv2.imwrite(str(person_output_path), cropped_person)

                except Exception as e:
                    logging.error(f"Error processing {ann_file}: {e}")
                    continue

        # Update progress
        with open(progress_file, 'a') as f:
            f.write(f"{image_name}\n")

    logging.info(f"Completed person extraction. Results saved to {output_dir}")

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    crop_persons_from_labels(
        labels_input_dir=YoloPaths.person_labels_input_dir,
        rawframe_dir=DetectionPaths.images_input_dir,
        output_dir=DetectionPaths.person_images_input_dir,
        progress_file=YoloPaths.person_extraction_progress_file_path,
        missing_frames_file=YoloPaths.person_missing_frames_file_path
    )

if __name__ == "__main__":    
    main()
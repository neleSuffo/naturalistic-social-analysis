import os
import cv2
import logging
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from constants import MtcnnPaths, DetectionPaths

def crop_faces_from_labels(
    labels_file: str = MtcnnPaths.labels_file_path,
    rawframe_dir: str = DetectionPaths.images_input_dir,
    output_dir: str = MtcnnPaths.faces_dir,
    labels_output_file: str = MtcnnPaths.gaze_labels_file_path,
    progress_file: str = MtcnnPaths.progress_file_path,
    missing_frames_file: str = MtcnnPaths.missing_frames_file_path,
):
    cv2.setNumThreads(1)
    
    # Convert output_dir to Path object and create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_dir = Path(progress_file).parent
    progress_dir.mkdir(parents=True, exist_ok=True)
    # Load progress
    processed_images = set()
    # Create progress file if it doesn't exist
# Initialize or load progress file
    try:
        if not progress_file.exists():
            progress_file.touch(mode=0o666)
            processed_images = set()
            logging.info(f"Created new progress file: {progress_file}")
        else:
            with open(progress_file, 'r') as f:
                processed_images = set(line.strip() for line in f)
            logging.info(f"Loaded {len(processed_images)} processed images from existing progress file")
    except Exception as e:
        logging.error(f"Error handling progress file: {e}")
        raise e

    with open(labels_file, 'r') as f, open(missing_frames_file, 'a') as missing_writer:
        for line in tqdm(f.readlines(), desc="Processing Labels"):
            parts = line.strip().split()
            image_name = parts[0]

            # Skip already processed images
            if image_name in processed_images:
                continue

            bbox_and_gaze = parts[1:]  # Extract bbox and gaze info
            video_name = "_".join(image_name.split("_")[:-1])
            frame_path = Path(rawframe_dir) / video_name / image_name

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
                bbox_values = list(map(float, values[:4]))
                gaze_label = int(values[4])

                x1, y1, w, h = bbox_values
                x2, y2 = x1 + w, y1 + h

                # Check bounding box validity
                if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                    logging.warning(f"Invalid bounding box for {frame_path}. Skipping.")
                    continue

                # Crop the face
                cropped_face = frame[int(y1):int(y2), int(x1):int(x2)]
                if cropped_face.size == 0:
                    logging.warning(f"Empty crop for {frame_path}. Skipping.")
                    continue

                # Save the cropped face
                cropped_face_name = f"{image_name.split('.')[0]}_face_{len(processed_images)}.jpg"
                cropped_face_path = output_dir / cropped_face_name
                cv2.imwrite(str(cropped_face_path), cropped_face)

                # Save label information
                with open(labels_output_file, 'a') as label_writer:
                    label_writer.write(f"{cropped_face_path} {gaze_label}\n")

            # Log processed image
            with open(progress_file, 'a') as progress_writer:
                progress_writer.write(f"{image_name}\n")
    
        logging.info(f"Cropped faces saved to {output_dir}")
        logging.info(f"Labels saved to {labels_output_file}")
        logging.info(f"Progress tracked in {progress_file}")
        logging.info(f"Missing frames logged in {missing_frames_file}")
    
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    crop_faces_from_labels()
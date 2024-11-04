import cv2
import logging
from pathlib import Path
from constants import FastReIDPaths, StrongSortPaths  
from config import StrongSortConfig


def save_cropped_images(
    video_file_name: Path
):
    """
    This function reads the detection file for a video and saves the cropped images in the appropriate directory.

    Parameters
    ----------
    video_file_name : Path
        the path to the video file folder containing the images and detection file
    """
    # Paths to your data
    images_dir = video_file_name / StrongSortConfig.image_subdir
    detection_file = video_file_name / StrongSortConfig.detection_file_path
    
    # Directory where cropped images will be saved
    output_dir = FastReIDPaths.base_dir / video_file_name.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the detection file
    with detection_file.open('r') as f:
        detections = f.readlines()

   # Iterate over each detection
    for line in detections:
        # Split the line by comma, and strip any whitespace around the values
        fields = line.strip().split(',')
        # Convert the frame_id to an integer (now safely handling commas)
        frame_id = int(fields[0].strip()) 
        # Bounding box (left, top, width, height)
        x, y, w, h = map(float, fields[2:6])

        # Load the corresponding frame image
        frame_path = images_dir / f'{frame_id:06d}.jpg'
        frame = cv2.imread(str(frame_path))

        # Check if the image exists
        if frame is None:
            print(f"Frame {frame_id} not found, skipping.")
            continue

        # Convert floating-point bounding box coordinates to integers for cropping
        x, y, w, h = map(int, [x * frame.shape[1], y * frame.shape[0], w * frame.shape[1], h * frame.shape[0]])
        
        # Crop the bounding box region from the frame
        cropped_img = frame[y:y+h, x:x+w]

        # Save the cropped image in the appropriate directory
        cropped_image_path = output_dir / f'{frame_id:06d}.jpg'
        cv2.imwrite(str(cropped_image_path), cropped_img)

    print(f"Cropping completed for {video_file_name} and images saved.")


def process_all_videos_in_folder(
    base_folder: Path
):
    """
    The function processes all videos in the given folder.

    Parameters
    ----------
    base_folder : Path
        the path to the folder containing the train and val directories
    """
    # Iterate over train and val directories
    for split in ['train', 'val']:
        split_folder = base_folder / split
        if split_folder.exists():
            # Iterate over each video folder inside train/ and val/ directories
            for video_folder in split_folder.iterdir():
                if video_folder.is_dir():  # Check if it's a directory
                    print(f"Processing {video_folder.name}...")
                    save_cropped_images(video_folder)
    
    # Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
def main():
    logging.info("Starting the prepare_fast_re_id_data script.")

    # Process all videos in the given folder
    process_all_videos_in_folder(StrongSortPaths.video_input_dir)

    logging.info("Finished the prepare_fast_re_id_data script.")

if __name__ == '__main__':
    main()
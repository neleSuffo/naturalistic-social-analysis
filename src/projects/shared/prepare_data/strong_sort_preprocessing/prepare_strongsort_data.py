from pathlib import Path
from ultralytics import YOLO
from src.projects.shared.utils import extract_frames_from_videos
from src.projects.social_interactions.common.constants import StrongSortPaths as SSP, YoloPaths as YP, ModelNames as MN  
from src.projects.social_interactions.config.config import StrongSortConfig as SSC, YoloConfig as YC
import logging


def run_detection_on_frames(
    base_folder: Path,
    ) -> None:
    """
    This function runs YOLOv8 detection on the frames in the given folder and saves the detections in the det folder.
    
    Parameters:
    ----------
    base_folder: Path
        The folder containing the video folders with the img1 and det subfolders.
    """
    # Load the YOLOv8 model
    model = YOLO(str(YP.trained_weights_path))

    # Process each video's img1 folder
    video_folders = [f for f in base_folder.iterdir() if f.is_dir()]

    for video_folder in video_folders:
        # Create the det folder if it doesn't exist
        img1_folder = video_folder / SSC.image_subdir
        det_file_path = video_folder / SSC.detection_file_path
        det_file_path.mkdir(parents=True, exist_ok=True)

        with det_file_path.open('w') as det_file:
            # Process each image in the img1 folder
            image_files = sorted(img1_folder.glob('*.jpg'))

            for image_file in image_files:
                frame_id = int(image_file.stem)
                # Run detection
                results = model(str(image_file), iou=YC.iou_threshold)

                for boxes in results[0].boxes:
                    # Extract the bounding box coordinates and confidence
                    x_center, y_center, width, height = boxes.xywhn[0]
                    conf = boxes.conf[0]
                    # Write the detection data to the det.txt file
                    det_file.write(f"{frame_id}, -1, {x_center.item():.2f}, {y_center.item():.2f}, {width.item():.2f}, {height.item():.2f}, {conf.item():.2f}, -1, -1, -1\n")

        logging.info(f"Detections saved for {video_folder.stem} in det.txt.")
        
    
def main():
    # Configure logging to show INFO level messages
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Step 1: Split the videos in the train and val and extract frames from them
    extract_frames_from_videos(SSP.video_input_dir, MN.strong_sort)

    # Step 3: Run YOLOv8 detection on the frames in the train and val folders
    run_detection_on_frames(SSP.train_videos_dir)
    run_detection_on_frames(SSP.val_videos_dir)

if __name__ == "__main__":
    main()
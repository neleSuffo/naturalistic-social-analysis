from pathlib import Path
from typing import Union
from ultralytics import YOLO
from src.projects.social_interactions.common.constants import StrongSortParameters as SSP, DetectionPaths as DP, YoloParameters as YP, TrainParameters as TP    
import cv2
import logging
import shutil
import random

def get_video_duration(
    video_path: Path
) -> float:
    """
    This function returns the duration of a video in seconds.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    cap.release()
    return duration

def split_videos_into_train_val() -> Union[list, list]:
    """
    This function splits the videos in the input folder into train and validation sets
    It returns the list of video names in the train and validation sets.
    """
    # Get all video files in the input folder
    input_folder = DP.videos_input
    train_ratio = TP.train_test_split
    video_files = [f for f in input_folder.iterdir() if f.is_file() and f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']]

    # Calculate total duration of all videos
    total_duration = sum(get_video_duration(video) for video in video_files)
    train_duration_target = total_duration * train_ratio

    # Shuffle videos to randomize assignment
    random.shuffle(video_files)

    # Initialize durations
    current_train_duration = 0
    train_videos = []
    val_videos = []

    # Assign videos to train and val
    for video in video_files:
        video_duration = get_video_duration(video)
        if current_train_duration + video_duration <= train_duration_target:
            train_videos.append(video.name)
            current_train_duration += video_duration
        else:
            val_videos.append(video.name)

    logging.info(f"Total duration: {total_duration:.2f} seconds")
    logging.info(f"Train set duration: {current_train_duration:.2f} seconds")
    logging.info(f"Validation set duration: {total_duration - current_train_duration:.2f} seconds")
    logging.info(f"Train videos: {len(train_videos)}")
    logging.info(f"Validation videos: {len(val_videos)}")

    return train_videos, val_videos
    
    
def extract_frames_from_videos(
    train_videos: list,
    val_videos: list,
    ) -> None:
    """
    This function extracts frames from all videos in the given folder and saves them as images
    in corresponding subfolders.
    
    Parameters:
    ----------
    train_videos: list
        The list of video names in the training set
    val_videos: list
        The list of video names in the validation set
        
    """
    # Set the input and output folders
    video_folder = DP.videos_input
    output_folder = SSP.video_input
    # Get all video files in the folder
    video_files = [f for f in video_folder.iterdir() if f.is_file() and f.suffix.lower() in ['.mp4', '.MP4']]

    # Process each video file
    for video_file in video_files:
        video_name = video_file.name

        if video_name in train_videos:
            output_video_folder = output_folder / 'train' / video_file.stem / 'img1'
        elif video_name in val_videos:
            output_video_folder = output_folder / 'val' / video_file.stem / 'img1'
        else:
            continue  # Skip videos not in train or val lists

        output_video_folder.mkdir(parents=True, exist_ok=True)

        # Open the video file
        cap = cv2.VideoCapture(str(video_file))
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Increment frame count and format the file name
            frame_count += 1
            frame_filename = f"{frame_count:06}.jpg"
            frame_filepath = output_video_folder / frame_filename

            # Save the frame as an image
            cv2.imwrite(str(frame_filepath), frame)

        cap.release()
        logging.info(f"Completed frame extraction for {video_name}.")

    logging.info(f"Completed frame extraction for all videos in {video_folder}.")


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
        img1_folder = video_folder / 'img1'
        det_folder = video_folder / 'det'
        det_folder.mkdir(parents=True, exist_ok=True)

        # Create the det.txt file
        det_file_path = det_folder / 'det.txt'

        with det_file_path.open('w') as det_file:
            # Process each image in the img1 folder
            image_files = sorted(img1_folder.glob('*.jpg'))

            for image_file in image_files:
                frame_id = int(image_file.stem)
                # Run detection
                results = model(str(image_file), iou=YP.iou_threshold)

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

    # Step 1: Split videos into train and val sets
    train_videos_list, val_videos_list = split_videos_into_train_val()

    # Step 2: Extract frames from the videos
    extract_frames_from_videos(train_videos_list, val_videos_list)

    # Step 3: Run YOLOv8 detection on the frames in the train and val folders
    run_detection_on_frames(SSP.videos_train)
    run_detection_on_frames(SSP.videos_val)
    

if __name__ == "__main__":
    main()
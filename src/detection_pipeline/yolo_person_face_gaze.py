import cv2
import logging
import argparse
import sqlite3
import pandas as pd
import numpy as np
import subprocess
from typing import Tuple
from pathlib import Path
from ultralytics import YOLO
from constants import YoloPaths, DetectionPaths, VTCPaths
from config import YoloConfig, DetectionPipelineConfig
from estimate_proximity import get_proximity

def extract_id_from_filename(filename: str) -> str:
    """
    This function extracts the ID from a filename.
    
    Parameters:
    ----------
    filename : str
        Filename to extract the ID from
    
    Returns:
    -------
    id_part : str
        Extracted ID part of the filename
    """
    parts = filename.split('id')
    if len(parts) > 1:
        id_part = parts[1].split('_')[0]
        return id_part
    return None

def get_balanced_videos(videos_dir: Path, age_df: pd.DataFrame, videos_per_group: int) -> list:
    """
    Select a balanced number of videos from each age group, skipping videos without age data.
    
    Parameters:
    ----------
    videos_dir : Path
        Directory containing the videos
    age_df : pd.DataFrame
        DataFrame with columns 'id' and 'age_group'
    videos_per_group : int
        Number of videos to select from each age group
        
    Returns:
    -------
    list
        List of selected video paths
    """
    selected_videos = []
    
    # Convert videos to a list with their IDs
    available_videos = []
    skipped_videos = []
    for video_path in videos_dir.glob("*.MP4"):
        video_id = extract_id_from_filename(video_path.stem)
        if video_id:
            try:
                # Check if ID exists in age_df and has a valid age group
                age_row = age_df[age_df['ID'] == int(video_id)]
                if not age_row.empty and pd.notna(age_row['Age Group'].iloc[0]):
                    available_videos.append((video_path, video_id))
                else:
                    skipped_videos.append(video_path.name)
            except (ValueError, KeyError):
                skipped_videos.append(video_path.name)
    
    if skipped_videos:
        logging.info(f"Skipped {len(skipped_videos)} videos without age data")
        logging.debug(f"Skipped videos: {', '.join(skipped_videos)}")
    
    # Group videos by age
    videos_by_age = {3: [], 4: [], 5: []}
    for video_path, video_id in available_videos:
        age_group = age_df[age_df['ID'] == int(video_id)]['Age Group'].iloc[0]
        if age_group in videos_by_age:
            videos_by_age[age_group].append(video_path)
    
    # Log available videos per age group
    for age_group, videos in videos_by_age.items():
        logging.info(f"Age group {age_group}: {len(videos)} videos available")
    
    # Select balanced number of videos from each group
    for age_group in videos_by_age:
        videos = videos_by_age[age_group]
        if len(videos) >= videos_per_group:
            selected = np.random.choice(videos, size=videos_per_group, replace=False)
            selected_videos.extend(selected)
            logging.info(f"Selected {len(selected)} videos from age group {age_group}")
        else:
            logging.warning(f"Age group {age_group} has only {len(videos)} videos, using all available")
            selected_videos.extend(videos)
    
    return selected_videos

def classify_gaze(gaze_model: YOLO, face_image: np.ndarray) -> Tuple[int, int]:
    """
    This function classifies the gaze direction of a face image.
    
    Parameters:
    ----------
    gaze_model : YOLO
        YOLO model for gaze classification
    face_image : np.ndarray
        Face image to be classified
        
    Returns:
    -------
    gaze_direction : int
        Gaze direction class label
    gaze_confidence : int
        Confidence score of the gaze classification
    """
    results = gaze_model(face_image)
    
    result = results[0].probs
    # Extract detection results
    object_cls = result.top1  # Class labels (0 = no_gaze, 1 = gaze)
    conf = result.top1conf.item()  # Confidence scores
    return object_cls, conf

def insert_video_record(video_path, cursor) -> int:
    """
    This function inserts a video record into the database if it doesn't already exist.
    It calculates the age at recording using the child's birthday from Subjects table.
    
    Parameters:
    ----------
    video_path : Path
        Path to the video file
    cursor : sqlite3.Cursor
        SQLite cursor object
    
    Returns:
    -------
    video_id : int
        Video ID
    """
    cursor.execute('SELECT video_id FROM Videos WHERE video_path = ?', (str(video_path),))
    existing_video = cursor.fetchone()
    
    if existing_video:
        logging.info(f"Video {Path(video_path)} already processed. Skipping.")
        return None
    else:
        # Extract child_id and recording_date from filename
        child_id, recording_date = extract_video_info(video_path)
                
        # Calculate age at recording using birthday from Subjects table
        age_at_recording = None
        if child_id and recording_date:
            # First get the birthday from Subjects table
            cursor.execute('''
                SELECT birthday 
                FROM Subjects 
                WHERE id = ?
            ''', (child_id,))
            result = cursor.fetchone()
        
            if result:
                birthday = result[0]
                # Now calculate age at recording
                cursor.execute('''
                    SELECT ROUND((julianday(?) - julianday(?))/365.25, 2)
                ''', (recording_date, birthday))
                age_result = cursor.fetchone()
                if age_result:
                    age_at_recording = age_result[0]
                else:
                    logging.warning(f"Could not calculate age for child ID {child_id}")
            else:
                logging.warning(f"No birthday found for child ID {child_id}")
            
        cursor.execute('''
            INSERT INTO Videos (video_path, child_id, recording_date, age_at_recording) 
            VALUES (?, ?, ?, ?)
        ''', (video_path, child_id, recording_date, age_at_recording))
        
        cursor.execute('SELECT video_id FROM Videos WHERE video_path = ?', (video_path,))
        return cursor.fetchone()[0]

def process_frame(frame: np.ndarray, frame_idx: int, video_id: int, detection_model: YOLO, gaze_model: YOLO, cursor: sqlite3.Cursor) -> dict:
    """
    This function processes a frame. It inserts the frame record and processes each object detected in the frame.
    The steps are as follows:
    1. Insert frame record
    2. Process each object detected in the frame
    
    Parameters:
    ----------
    frame : np.ndarray
        Frame to be processed
    frame_idx : int
        Frame index
    video_id : int
        Video ID
    detection_model : YOLO
        YOLO model for person, face, and object detection
    gaze_model : YOLO
        YOLO model for gaze classification
    cursor : sqlite3.Cursor
        SQLite cursor object
    
    Returns:
    -------
    detection_counts : dict
        Dictionary containing counts of each object class detected
    """
    cursor.execute('INSERT INTO Frames (video_id, frame_number) VALUES (?, ?)', (video_id, frame_idx))

    results = detection_model.predict(frame, iou=YoloConfig.best_iou)[0]
    detection_counts = {name: 0 for name in YoloConfig.detection_mapping.values()}

    # Extract detection results
    conf = results.boxes.conf  # Confidence scores
    object_cls = results.boxes.cls  # Class labels
    xyxy = results.boxes.xyxy  # Bounding boxes in xyxy format (x_min, y_min, x_max, y_max)

    # Process each detection
    for box in results.boxes:
        # Extract coordinates, class and confidence
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # Convert tensor to numpy
        class_id = int(box.cls[0].cpu().numpy())  # Convert tensor to numpy
        confidence_score = float(box.conf[0].cpu().numpy())  # Convert tensor to numpy
        class_name = detection_model.names[class_id]

        # Update detection count
        if class_name in detection_counts:
            detection_counts[class_name] += 1

        # Initialize face-specific variables
        gaze_direction = None
        gaze_confidence = None
        proximity = None

        # Process faces for gaze and proximity
        if class_id in [2, 3]:  # child or adult face
            bounding_box = [x1, y1, x2, y2]
            face_image = frame[y1:y2, x1:x2]
            gaze_direction, gaze_confidence = classify_gaze(gaze_model, face_image)
            proximity = get_proximity(bounding_box, class_name)

        # Insert detection record
        cursor.execute('''
            INSERT INTO Detections
            (video_id, frame_number, object_class, confidence_score,
            x_min, y_min, x_max, y_max, gaze_direction,
            gaze_confidence, proximity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (video_id, frame_idx, class_id, confidence_score, x1, y1, x2, y2, gaze_direction, gaze_confidence, proximity))

    return detection_counts

def process_video(video_folder: Path, 
                  detection_model: YOLO, 
                  gaze_model: YOLO, 
                  cursor: sqlite3.Cursor, 
                  conn: sqlite3.Connection,
                  frame_skip: int):
    """
    Process frames from saved image files instead of video.
    """
    logging.info(f"Processing video: {video_folder}")
    
    # Convert Path to string for SQLite
    video_folder_str = str(video_folder)
    
    # Check if video exists in VideoStatistics table
    cursor.execute('''
        SELECT v.video_id, vs.processed_frames 
        FROM Videos v
        LEFT JOIN VideoStatistics vs ON v.video_id = vs.video_id
        WHERE v.video_path = ?
    ''', (video_folder_str,))
    result = cursor.fetchone()
    
    if result and result[1] is not None and result[1] > 0:  # Check if frames were actually processed
        logging.info(f"Video {video_folder} already processed with {result[1]} frames. Skipping.")
        return
    
    # Continue with video processing if not already processed
    video_id = insert_video_record(video_folder_str, cursor)
    
    # Skip if video already processed
    if video_id is None:
        return

    # Get corresponding frames directory
    frames_dir = DetectionPaths.images_input_dir / video_folder
    
    if not frames_dir.exists():
        logging.error(f"Frames directory not found: {frames_dir}")
        return

    # Get all frame images sorted by frame number
    frame_files = sorted(frames_dir.glob('*.jpg'))
    processed_frames = 0
    total_counts = {name: 0 for name in YoloConfig.detection_mapping.values()}

    for frame_file in frame_files:
        # Extract frame number from filename (last 6 digits before .jpg)
        padded_frame_idx = frame_file.stem[-6:]  # e.g. "000533"
        frame_idx = int(padded_frame_idx)  # automatically converts to 533

        # Skip frames according to frame_skip
        if frame_skip > 0 and frame_idx % frame_skip != 0:
            continue
            
        # Read frame
        frame = cv2.imread(str(frame_file))
        if frame is None:
            logging.warning(f"Failed to read frame: {frame_file}")
            continue

        # Process frame
        detection_counts = process_frame(
            frame, frame_idx, video_id, detection_model, gaze_model, cursor
        )
        
        # Update total counts
        for class_name, count in detection_counts.items():
            total_counts[class_name] += count
            
        conn.commit()
        processed_frames += 1

    # Store statistics in database
    cursor.execute('''
        INSERT INTO VideoStatistics (
            video_id, total_frames, processed_frames,
            child_count, adult_count, child_face_count, adult_face_count,
            book_count, toy_count, kitchenware_count, screen_count,
            food_count, other_object_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        video_id, len(frame_files), processed_frames,
        total_counts['infant/child'], total_counts['adult'],
        total_counts['infant/child face'], total_counts['adult face'],
        total_counts['book'], total_counts['toy'],
        total_counts['kitchenware'], total_counts['screen'],
        total_counts['food'], total_counts['other_object']
    ))
    conn.commit()
    
    logging.info(f"Processed {processed_frames} frames out of {len(frame_files)} total frames")

def extract_video_info(video_path: str) -> Tuple[int, str]:
    """
    Extracts child ID and recording date from video filename.
    Expected format: *id{number}_{YYYY_MM_DD}*.MP4
    
    Parameters:
    ----------
    video_path : str
        Video file path or name
        
    Returns:
    -------
    tuple : (child_id, recording_date)
        child_id : int or None
        recording_date : str or None
    """
    try:
        # Convert Path to string if necessary
        video_name = str(video_path)
        if isinstance(video_path, Path):
            video_name = video_path.name
            
        # Extract ID
        id_start = video_name.find('id') + 2
        if id_start < 2:  # if 'id' not found
            return None, None
            
        id_end = video_name[id_start:].find('_') + id_start
        child_id = int(video_name[id_start:id_end])
        
        # Extract date
        date_start = video_name.find('_20') + 1
        if date_start < 1:  # if '_20' not found
            return child_id, None
            
        recording_date = video_name[date_start:date_start+10].replace('_', '-')
        
        return child_id, recording_date
    except (ValueError, IndexError):
        return None, None
    
def store_voice_detections(video_file_name: str, results_file: Path, fps: int = 30):
    """
    Reads the voice type classifier RTTM output and stores detections per video frame.
    For each detection, infers the corresponding frames (using fps) and inserts a row for each.
    
    Parameters:
    ----------
    video_file_name : str
        The video file name used as the video_path in the Videos table.
    results_file : Path
        Path to the RTTM results file.
    fps : int
        Frames per second of the video (default: 30).
    """
    with sqlite3.connect(DetectionPaths.detection_db_path) as conn:
        cursor = conn.cursor()
    
        # Insert video record if it does not exist
        child_id, recording_date = extract_video_info(video_file_name)
        cursor.execute('''
            INSERT OR IGNORE INTO Videos (video_path, child_id, recording_date) 
            VALUES (?, ?, ?)
        ''', (video_file_name, child_id, recording_date))
        cursor.execute("SELECT video_id FROM Videos WHERE video_path = ?", (video_file_name,))
        video_id = cursor.fetchone()[0]
        
        with open(results_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                # RTTM format:
                # SPEAKER <audio_id> <channel> <start_time> <duration> <NA> <NA> <label> <NA> <NA>
                try:
                    start_time = float(parts[3])
                    duration = float(parts[4])
                except ValueError:
                    continue
                end_time = start_time + duration
                object_class_str = parts[7]  # e.g., "KCHI", "FEM", etc.
                
                # Map object class to integer and store into classes if not already present.
                cursor.execute("SELECT class_id FROM Classes WHERE class_name = ?", (object_class_str,))
                result_class = cursor.fetchone()
                if result_class is None:
                    cursor.execute("INSERT INTO Classes (class_name) VALUES (?)", (object_class_str,))
                    class_id = cursor.lastrowid
                else:
                    class_id = result_class[0]            
                
                # For example, if detection starts at frame 8:
                start_frame = int(start_time * fps)
                # And number of affected frames based on duration:
                num_frames = int(duration * fps)
                
                for frame_offset in range(num_frames):
                    actual_frame = start_frame + frame_offset
                    # Insert frame record if it doesn't exist
                    cursor.execute('INSERT OR IGNORE INTO Frames (video_id, frame_number) VALUES (?, ?)', 
                                (video_id, actual_frame))
                                        
                    result = cursor.fetchone()
                    if result is None:
                        cursor.execute("INSERT INTO Frames (video_id, frame_number) VALUES (?, ?)", (video_id, actual_frame))
                        frame_number = cursor.lastrowid
                    else:
                        frame_number = result[0]
                    
                    # Insert a detection for this specific frame.
                    # For audio there are no spatial coordinates, so we set them to 0.
                    # Insert detection record
                    cursor.execute('''
                        INSERT INTO Detections 
                        (video_id, frame_number, object_class, confidence_score, x_min, y_min, x_max, y_max, gaze_direction, gaze_confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (video_id, actual_frame, object_class_str, None, None, None, None, None, None, None))
        
        conn.commit()
        conn.close()
        logging.info("Voice detections stored in the database.")
    
    
def run_voice_type_classifier(video_file_name):
    """
    This function runs the voice type classifier on the given video.
    
    Parameters:
    ----------
    video_path : Path
        Path to the video file
    """
    # get corresponding audio file
    audio_file_name = video_file_name.replace(".MP4", "_16kHz.wav")
    audio_file_path = VTCPaths.quantex_output_folder / audio_file_name
     # Run voice type classifier using apply.sh.
    # This command changes to the 'projects/voice-type-classifier' directory, activates the pyannote conda environment,
    # and executes the apply.sh script.
    voice_command = (
        "cd /home/nele_pauline_suffo/projects/voice-type-classifier && " 
        f"conda run -n pyannote ./apply.sh {audio_file_path} --device=gpu"
    )
    subprocess.run(voice_command, shell=True, check=True)
    
    # Assume the classifier writes its output to an 'all.rttm' file in the designated results directory.
    vtc_results_dir = VTCPaths.vtc_results_dir
    rttm_file = vtc_results_dir / video_file_name / "all.rttm"
    
    if rttm_file.exists():
        store_voice_detections(video_file_name, rttm_file)
    else:
        logging.error(f"RTTM results file not found: {rttm_file}")
 
def register_model(cursor: sqlite3.Cursor, model_name: str, model: YOLO) -> int:
    """
    Registers a YOLO model and its classes in the database.
    
    Parameters:
    ----------
    cursor : sqlite3.Cursor
        The SQLite database cursor.
    model_name : str
        The name of the model.
    model : YOLO
        The YOLO model instance which contains a `model.names` mapping.
        
    Returns:
    -------
    model_id : int
        The model_id assigned in the database.
    """
    # Insert model into Models table if not exists and retrieve its model_id
    cursor.execute("INSERT OR IGNORE INTO Models (model_name) VALUES (?)", (model_name,))
    cursor.execute("SELECT model_id FROM Models WHERE model_name = ?", (model_name,))
    model_id = cursor.fetchone()[0]
    
    # Insert associated classes into the Classes table
    for class_id, class_name in model.model.names.items():
        cursor.execute('''
            INSERT OR IGNORE INTO Classes (class_id, model_id, class_name)
            VALUES (?, ?, ?)
        ''', (class_id, model_id, class_name))
        
    return model_id
 
def update_detection_summary(cursor: sqlite3.Cursor, conn: sqlite3.Connection):
    """
    Updates the DetectionSummary table with aggregated detection statistics.
    """
    summary_query = '''
    WITH FramesWithDetections AS (
        SELECT 
            v.child_id as id,
            v.video_id,
            d.frame_number,
            MAX(CASE WHEN d.object_class = 1.0 THEN 1 ELSE 0 END) as has_adult,
            MAX(CASE WHEN d.object_class = 0.0 THEN 1 ELSE 0 END) as has_child,
            MAX(CASE WHEN d.object_class = 3.0 THEN 1 ELSE 0 END) as has_adult_face,
            MAX(CASE WHEN d.object_class = 2.0 THEN 1 ELSE 0 END) as has_child_face,
            MAX(CASE WHEN d.object_class = 5.0 THEN 1 ELSE 0 END) as has_book,
            MAX(CASE WHEN d.object_class = 6.0 THEN 1 ELSE 0 END) as has_toy,
            MAX(CASE WHEN d.object_class = 7.0 THEN 1 ELSE 0 END) as has_kitchenware,
            MAX(CASE WHEN d.object_class = 8.0 THEN 1 ELSE 0 END) as has_screen,
            MAX(CASE WHEN d.object_class = 9.0 THEN 1 ELSE 0 END) as has_food,
            MAX(CASE WHEN d.object_class = 10.0 THEN 1 ELSE 0 END) as has_other_object
        FROM Videos v
        JOIN Detections d ON v.video_id = d.video_id
        GROUP BY v.child_id, v.video_id, d.frame_number
    ),
    TotalFramesPerChild AS (
        SELECT 
            v.child_id as id,
            SUM(vs.total_frames) as total_frames
        FROM Videos v
        JOIN VideoStatistics vs ON v.video_id = vs.video_id
        GROUP BY v.child_id
    )
    INSERT OR REPLACE INTO DetectionSummary
    SELECT 
        f.id,
        COUNT(DISTINCT f.video_id) as video_count,
        t.total_frames,
        SUM(f.has_adult) as frames_with_adult,
        SUM(f.has_child) as frames_with_child,
        SUM(f.has_adult_face) as frames_with_adult_face,
        SUM(f.has_child_face) as frames_with_child_face,
        SUM(f.has_book) as frames_with_book,
        SUM(f.has_toy) as frames_with_toy,
        SUM(f.has_kitchenware) as frames_with_kitchenware,
        SUM(f.has_screen) as frames_with_screen,
        SUM(f.has_food) as frames_with_food,
        SUM(f.has_other_object) as frames_with_other_object,
        ROUND(SUM(f.has_adult) * 100.0 / t.total_frames, 2) as adult_percent,
        ROUND(SUM(f.has_child) * 100.0 / t.total_frames, 2) as child_percent,
        ROUND(SUM(f.has_adult_face) * 100.0 / t.total_frames, 2) as adult_face_percent,
        ROUND(SUM(f.has_child_face) * 100.0 / t.total_frames, 2) as child_face_percent,
        ROUND(SUM(f.has_book) * 100.0 / t.total_frames, 2) as book_percent,
        ROUND(SUM(f.has_toy) * 100.0 / t.total_frames, 2) as toy_percent,
        ROUND(SUM(f.has_kitchenware) * 100.0 / t.total_frames, 2) as kitchenware_percent,
        ROUND(SUM(f.has_screen) * 100.0 / t.total_frames, 2) as screen_percent,
        ROUND(SUM(f.has_food) * 100.0 / t.total_frames, 2) as food_percent,
        ROUND(SUM(f.has_other_object) * 100.0 / t.total_frames, 2) as other_object_percent
    FROM FramesWithDetections f
    JOIN TotalFramesPerChild t ON f.id = t.id
    GROUP BY f.id, t.total_frames;
    '''
    cursor.execute(summary_query)
    conn.commit()
          
def main(num_videos_to_process: int = None,
        frame_skip: int = 10):
    """
    This function processes videos using YOLO models for person and face detection and gaze classification.
    It loads the age group data and processes either all videos or a balanced subset from each age group.
    
    Parameters:
    ----------
    num_videos_to_process : int, optional
        Number of videos to process per age group. If None, processes all available videos.
    frame_skip : int, optional
        Number of frames to skip between processing (default: 10)
    """
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load the age group data
    age_df = pd.read_csv('/home/nele_pauline_suffo/ProcessedData/age_group.csv')
    
    videos_input_dir = DetectionPaths.images_input_dir
    conn = sqlite3.connect(DetectionPaths.detection_db_path)
    cursor = conn.cursor()
    
    # Initialize models once
    detection_model = YOLO(YoloPaths.all_trained_weights_path)
    gaze_model = YOLO(YoloPaths.gaze_trained_weights_path)   
    
    detection_model_id = register_model(cursor, "detection", detection_model)
    gaze_model_id = register_model(cursor, "gaze", gaze_model)

    videos_to_not_process = DetectionPipelineConfig.videos_to_not_process
    if num_videos_to_process is None:
        # Process all video folders
        all_videos = [p for p in videos_input_dir.iterdir() if p.is_dir()]
        videos_to_process = [v for v in all_videos if v.name not in videos_to_not_process]

        skipped = len(all_videos) - len(videos_to_process)
        logging.info(f"Found {len(all_videos)} video folders")
        logging.info(f"Skipping {skipped} videos from exclusion list")
        logging.info(f"Processing {len(videos_to_process)} videos")
    else:
        # Get balanced videos across age groups
        videos_per_group = num_videos_to_process // 3
        all_videos = get_balanced_videos(videos_input_dir, age_df, videos_per_group=videos_per_group)
        videos_to_process = [v for v in all_videos if v.name not in videos_to_not_process]
        
        skipped = len(all_videos) - len(videos_to_process)
        logging.info(f"Selected {len(all_videos)} balanced videos")
        logging.info(f"Skipping {skipped} videos from exclusion list")
        logging.info(f"Processing {len(videos_to_process)} videos")
    
    # Process videos
    processed_children = set()
    for video in videos_to_process:
        process_video(video, detection_model, gaze_model, cursor, conn, frame_skip)
        #run_voice_type_classifier(video_path.name)

        # Get child_id from the video that was just processed
        child_id, _ = extract_video_info(video)
        if child_id and child_id not in processed_children:
            # Update summary only for this child's data
            update_detection_summary(cursor, conn)
            processed_children.add(child_id)
            logging.info(f"Updated detection summary for child {child_id}")
    
    conn.close()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Process a set of videos using YOLO models for person and face detection and gaze classification.")
    argparser.add_argument("--num_videos", type=int, help="Number of videos to process")
    argparse.add_argument("--frame_skip", type=int, default=10, help="Number of frames to skip between processing (default: 10)")
    args = argparser.parse_args()
    num_videos_to_process = args.num_videos
    frame_skip = args.frame_skip
    main(num_videos_to_process, frame_skip)
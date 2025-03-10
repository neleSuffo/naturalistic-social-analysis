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
        logging.info(f"Video {Path(video_path).name} already processed. Skipping.")
        return None
    else:
        cursor.execute('INSERT INTO Videos (video_path) VALUES (?)', (str(video_path),))
        cursor.execute('SELECT video_id FROM Videos WHERE video_path = ?', (str(video_path),))
        return cursor.fetchone()[0]

def process_frame(frame: np.ndarray, frame_idx: int, video_id: int, person_face_model: YOLO, gaze_model: YOLO, cursor: sqlite3.Cursor) -> Tuple[int, int]:
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
    person_face_model : YOLO
        YOLO model for person and face detection
    gaze_model : YOLO
        YOLO model for gaze classification
    cursor : sqlite3.Cursor
        SQLite cursor object
    
    Returns:
    -------
    num_child : int
        Number of children detected
    num_adult : int
        Number of adults detected
    num_child_faces : int
        Number of child faces detected
    num_adult_faces : int
        Number of adult faces detected
    """
    cursor.execute('INSERT INTO Frames (video_id, frame_number) VALUES (?, ?)', (video_id, frame_idx))

    result = person_face_model(frame)
    num_child, num_adult, num_child_faces, num_adult_faces = 0, 0, 0, 0

    results = result[0].boxes
    # Extract detection results
    conf = results.conf  # Confidence scores
    object_cls = results.cls  # Class labels
    xyxy = results.xyxy  # Bounding boxes in xyxy format (x_min, y_min, x_max, y_max)

    # Iterate over each detection
    for i in range(len(conf)):
        # Extract bounding box and class label
        x_min, y_min, x_max, y_max = map(int, xyxy[i])  # Bounding box coordinates
        confidence_score = conf[i].item()  # Confidence score
        object_class = object_cls[i].item()  # Class label

        # Determine if the detected object is a person or a face
        if object_class == 0: # infant/child
            num_child += 1
            gaze_direction = None
            gaze_confidence = None
        elif object_class == 1:  # adult
            num_adult += 1
            gaze_direction = None
            gaze_confidence = None
        elif object_class == 2:  # child face
            num_child_faces += 1
            face_image = frame[y_min:y_max, x_min:x_max]
            gaze_direction, gaze_confidence = classify_gaze(gaze_model, face_image)
        elif object_class == 3:  # adult face
            num_adult_faces += 1
            face_image = frame[y_min:y_max, x_min:x_max]
            gaze_direction, gaze_confidence = classify_gaze(gaze_model, face_image)
        else:
            continue  # Skip if the object is neither 'person' nor 'face'

        # Insert detection record into the database
        cursor.execute('''
            INSERT INTO Detections 
            (video_id, frame_number, object_class, confidence_score, x_min, y_min, x_max, y_max, gaze_direction, gaze_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (video_id, frame_idx, object_class, confidence_score, x_min, y_min, x_max, y_max, gaze_direction, gaze_confidence))

    return num_child, num_adult, num_child_faces, num_adult_faces

def process_video(video_path: Path, person_face_model: YOLO, gaze_model: YOLO, cursor: sqlite3.Cursor, conn: sqlite3.Connection):
    """
    This function processes a video frame by frame. It inserts the video record, processes each frame, and commits the changes.
    The steps are as follows:
    1. Insert video record
    2. Open video capture
    3. Process each frame
    4. Commit changes
    5. Close video capture
    
    Parameters:
    ----------
    video_path : Path
        Path to the video file
    person_face_model : YOLO
        YOLO model for person and face detection
    gaze_model : YOLO
        YOLO model for gaze classification
    cursor : sqlite3.Cursor
        SQLite cursor object
    conn : sqlite3.Connection
        SQLite connection object
    """
    logging.info(f"Processing video: {video_path.name}")
    video_id = insert_video_record(video_path.name, cursor)
    
    # Skip if video already processed
    if video_id is None:
        return
        
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    total_children, total_adults, total_child_faces, total_adult_faces = 0, 0, 0, 0
    total_gaze, total_no_gaze = 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 10 == 0:
            num_child, num_adult, num_child_faces, num_adult_faces = process_frame(
                frame, frame_idx, video_id, person_face_model, gaze_model, cursor
            )
            total_children += num_child
            total_adults += num_adult
            total_child_faces += num_child_faces
            total_adult_faces += num_adult_faces
            conn.commit()

        frame_idx += 1

    cap.release()
    
    # Create output directory if it doesn't exist
    output_dir = Path(DetectionPaths.detection_results_dir) / "stats"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create statistics file
    output_file = output_dir / f"{video_path.stem}_stats.txt"
    with open(output_file, 'w') as f:
        f.write(f"Video Statistics for: {video_path.name}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Video ID: {video_id}\n")
        f.write(f"Total Frames Processed: {frame_idx}\n")
        f.write("\nDetection Counts:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Children: {total_children}\n")
        f.write(f"Adults: {total_adults}\n")
        f.write(f"Child Faces: {total_child_faces}\n")
        f.write(f"Adult Faces: {total_adult_faces}\n")
    
    logging.info(f"Statistics written to: {output_file}")
    conn.commit()

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
    conn = sqlite3.connect(DetectionPaths.detection_db_path)
    cursor = conn.cursor()
    
    # Insert video record if it does not exist
    cursor.execute("INSERT OR IGNORE INTO Videos (video_path) VALUES (?)", (video_file_name,))
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
       
def main(num_videos_to_process: int):
    """
    This function processes a set of videos using YOLO models for person and face detection and gaze classification.
    It loads the age group data, selects a balanced number of videos from each age group, and processes each video.
    
    Parameters:
    ----------
    num_videos_to_process : int
        Number of videos to process
    """
    videos_per_group = num_videos_to_process // 3
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load the age group data
    age_df = pd.read_csv('/home/nele_pauline_suffo/ProcessedData/age_group.csv')
    
    videos_input_dir = DetectionPaths.quantex_videos_input_dir
    conn = sqlite3.connect(DetectionPaths.detection_db_path)
    cursor = conn.cursor()
    
    # Initialize models once
    detection_model = YOLO(YoloPaths.all_trained_weights_path)
    gaze_model = YOLO(YoloPaths.gaze_trained_weights_path)   
    
    detection_model_id = register_model(cursor, "detection", detection_model)
    gaze_model_id = register_model(cursor, "gaze", gaze_model)

    # Get balanced videos across age groups (20 from each group)
    selected_videos = get_balanced_videos(videos_input_dir, age_df, videos_per_group=videos_per_group)
    
    # Process only the selected videos
    for video_path in selected_videos:
        process_video(video_path, detection_model, gaze_model, cursor, conn)
        #run_voice_type_classifier(video_path.name)
    
    conn.close()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Process a set of videos using YOLO models for person and face detection and gaze classification.")
    argparser.add_argument("--num_videos", type=int, help="Number of videos to process")
    args = argparser.parse_args()
    num_videos_to_process = args.num_videos_to_process
    main()
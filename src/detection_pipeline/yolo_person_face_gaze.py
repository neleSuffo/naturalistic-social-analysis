import cv2
import logging
import sqlite3
from pathlib import Path
from ultralytics import YOLO
from constants import YoloPaths, DetectionPaths

def classify_gaze(face_image):
    """
    Placeholder for gaze classification.
    Return 0 if gaze is not directed at the camera, 1 if it is.
    """
    gaze_model = YOLO(YoloPaths.gaze_trained_weights_path)
    result = gaze_model(face_image)
    
    results = result[0].boxes
    # Extract detection results
    conf = results.conf  # Confidence scores
    object_cls = results.cls  # Class labels
    
    #return conf[0], object_cls[0]
    return 0, 0

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

def process_frame(frame, frame_idx, video_id, model, cursor) -> tuple:
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
    model : YOLO
        YOLO model for object detection
    cursor : sqlite3.Cursor
        SQLite cursor object
    
    Returns:
    -------
    num_persons : int
        Number of persons detected in the frame
    num_faces : int
        Number of faces detected in the frame
    """
    cursor.execute('INSERT INTO Frames (video_id, frame_number) VALUES (?, ?)', (video_id, frame_idx))
    frame_id = cursor.lastrowid

    result = model(frame)
    num_persons = 0
    num_faces   = 0

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
        if object_class == 0: # person
            num_persons += 1
            gaze_direction = None
        # TODO: needs adjustment for three classes
        elif object_class == 1:  # child body parts
            num_faces += 1
            face_image = frame[y_min:y_max, x_min:x_max]
            gaze_direction, gaze_confidence = classify_gaze(face_image)
        else:
            continue  # Skip if the object is neither 'person' nor 'face'

        # Insert detection record into the database
        cursor.execute('''
            INSERT INTO Detections 
            (frame_id, object_class, confidence_score, x_min, y_min, x_max, y_max, gaze_direction, gaze_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (frame_id, object_class, confidence_score, x_min, y_min, x_max, y_max, gaze_direction, gaze_confidence))

    return num_persons, num_faces

def process_video(video_path, model, cursor, conn):
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
    model : YOLO
        YOLO model for object detection
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
    total_persons = 0
    total_faces = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 10 == 0:
            num_persons, num_faces = process_frame(frame, frame_idx, video_id, model, cursor)
            total_persons += num_persons
            total_faces += num_faces
            conn.commit()  # Commit changes after processing each frame

        frame_idx += 1

    cap.release()
    logging.info(f"Finished video: {video_path} | Persons: {total_persons} | Faces: {total_faces}")
    conn.commit()  # Commit changes after processing the entire video

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
            object_class = parts[7]  # e.g. KCHI, FEM, etc.
            
            # For example, if detection starts at frame 8:
            start_frame = int(start_time * fps)
            # And number of affected frames based on duration:
            num_frames = int(duration * fps)
            
            for frame_offset in range(num_frames):
                actual_frame = start_frame + frame_offset
                # Check if a frame record exists; if not, create one.
                cursor.execute("SELECT frame_number FROM Frames WHERE video_id = ? AND frame_number = ?", (video_id, actual_frame))
                result = cursor.fetchone()
                if result is None:
                    cursor.execute("INSERT INTO Frames (video_id, frame_number) VALUES (?, ?)", (video_id, actual_frame))
                    frame_number = cursor.lastrowid
                else:
                    frame_number = result[0]
                
                # Insert a detection for this specific frame.
                # For audio there are no spatial coordinates, so we set them to 0.
                cursor.execute('''
                    INSERT INTO Detections 
                    (frame_number, object_class, confidence_score, x_min, y_min, x_max, y_max, gaze_direction, gaze_confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (frame_id, object_class, None, None, None, None, None, None, None))
    
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
    rttm_file = vtc_results_dir / "all.rttm"
    
    if rttm_file.exists():
        store_voice_detections(video_file_name, rttm_file)
    else:
        logging.error(f"RTTM results file not found: {rttm_file}")
        
def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    videos_input_dir = DetectionPaths.quantex_videos_input_dir
    conn = sqlite3.connect(DetectionPaths.detection_db_path)
    cursor = conn.cursor()
    
    yolo_model = YOLO(YoloPaths.person_trained_weights_path)

    # get model class names as dictionary with id and name
    yolo_classes = yolo_model.model.names
    for class_id, class_name in yolo_classes.items():
        cursor.execute('''
            INSERT OR IGNORE INTO YOLOClasses (class_id, class_name)
            VALUES (?, ?)
        ''', (class_id, class_name))
    
    for video_path in videos_input_dir.glob("*.MP4"):
        process_video(video_path, yolo_model, cursor, conn)
        run_voice_type_classifier(video_path.name)
        return
    conn.close()

if __name__ == "__main__":
    main()
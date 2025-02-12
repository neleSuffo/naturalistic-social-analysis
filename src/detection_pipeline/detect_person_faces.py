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
    return 0

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
            gaze_direction = 0  # Placeholder for gaze classification
        else:
            continue  # Skip if the object is neither 'person' nor 'face'

        # Insert detection record into the database
        cursor.execute('''
            INSERT INTO Detections 
            (frame_id, object_class, confidence_score, x_min, y_min, x_max, y_max, gaze_direction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (frame_id, object_class, confidence_score, x_min, y_min, x_max, y_max, gaze_direction))

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
        return
    conn.close()

if __name__ == "__main__":
    main()
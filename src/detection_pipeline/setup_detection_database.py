import sqlite3
from pathlib import Path
from constants import DetectionPaths

def setup_detection_database(db_path: Path = DetectionPaths.detection_db_path):
    """
    This function sets up the SQLite database for storing detection results.
    
    Parameters:
    ----------
    db_path : Path
        Path to the SQLite database file, defaults to DetectionPaths.detection_db_path
    """
    # Ensure directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Videos (
            video_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_path TEXT UNIQUE
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Frames (
            frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            frame_number INTEGER,
            FOREIGN KEY (video_id) REFERENCES Videos(video_id)
        )
    ''')

    # Create YOLOClasses table to map class numbers to class names
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS YOLOClasses (
            class_id INTEGER PRIMARY KEY,
            class_name TEXT UNIQUE
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Detections (
            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_number INTEGER,
            object_class TEXT,
            confidence_score REAL,
            x_min INTEGER,
            y_min INTEGER,
            x_max INTEGER,
            y_max INTEGER,
            gaze_direction INTEGER,
            gaze_confidence REAL,
            FOREIGN KEY (frame_number) REFERENCES Frames(frame_number)
        )
    ''')

    conn.commit()
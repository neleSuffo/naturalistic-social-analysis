import sqlite3
import logging
from pathlib import Path
from constants import DetectionPaths

logging.basicConfig(level=logging.INFO)

def setup_detection_database(db_path: Path = DetectionPaths.detection_db_path):
    """
    This function sets up the SQLite database for storing detection results.
    If the database already exists, it skips creation.
    
    Parameters:
    ----------
    db_path : Path
        Path to the SQLite database file, defaults to DetectionPaths.detection_db_path
    """
    # Check if database already exists
    if db_path.exists():
        logging.info(f"Database already exists at {db_path}. Skipping creation.")
        return
    
    # Ensure directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Connect to SQLite database (create it since we know it doesn't exist)
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
            UNIQUE(video_id, frame_number),
            FOREIGN KEY (video_id) REFERENCES Videos(video_id)
        )
    ''')


    cursor.execute('''
        CREATE TABLE Models (
            model_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT UNIQUE
        )
    ''')

    cursor.execute('''
        CREATE TABLE Classes (
            class_id INTEGER,
            model_id INTEGER,
            class_name TEXT,
            PRIMARY KEY (model_id, class_name),
            FOREIGN KEY (model_id) REFERENCES Models(model_id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Detections (
            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            frame_number INTEGER,
            object_class TEXT,
            confidence_score REAL,
            x_min INTEGER,
            y_min INTEGER,
            x_max INTEGER,
            y_max INTEGER,
            gaze_direction INTEGER,
            gaze_confidence REAL,
            FOREIGN KEY (video_id, frame_number) REFERENCES Frames(video_id, frame_number)
        )
    ''')

    conn.commit()
    conn.close()
    logging.info(f"Detection database created at {db_path}")
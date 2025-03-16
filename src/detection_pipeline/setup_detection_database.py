import sqlite3
import logging
import pandas as pd
from pathlib import Path
from constants import DetectionPaths, DetectionPipeline

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
            video_path TEXT UNIQUE,
            child_id INTEGER,
            recording_date DATE,
            age_at_recording FLOAT
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
            proximity REAL,
            FOREIGN KEY (video_id, frame_number) REFERENCES Frames(video_id, frame_number)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS VideoStatistics (
            video_id INTEGER PRIMARY KEY,
            total_frames INTEGER,
            processed_frames INTEGER,
            child_count INTEGER,
            adult_count INTEGER,
            child_face_count INTEGER,
            adult_face_count INTEGER,
            book_count INTEGER,
            toy_count INTEGER,
            kitchenware_count INTEGER,
            screen_count INTEGER,
            food_count INTEGER,
            other_object_count INTEGER,
            FOREIGN KEY (video_id) REFERENCES Videos(video_id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS DetectionSummary (
            id INTEGER PRIMARY KEY,
            video_count INTEGER,
            total_frames INTEGER,
            frames_with_adult INTEGER,
            frames_with_child INTEGER,
            frames_with_adult_face INTEGER,
            frames_with_child_face INTEGER,
            frames_with_book INTEGER,
            frames_with_toy INTEGER,
            frames_with_kitchenware INTEGER,
            frames_with_screen INTEGER,
            frames_with_food INTEGER,
            frames_with_other_object INTEGER,
            adult_percent REAL,
            child_percent REAL,
            adult_face_percent REAL,
            child_face_percent REAL,
            book_percent REAL,
            toy_percent REAL,
            kitchenware_percent REAL,
            screen_percent REAL,
            food_percent REAL,
            other_object_percent REAL
        )
    ''')

    # Load subjects data from CSV with header row
    quantex_subjects_df = pd.read_csv(
        DetectionPipeline.quantex_subjects.with_suffix('.csv'),
        header=0,  # Use first row as column names
        sep=';',   # Specify semicolon as separator
        encoding='utf-8'  # Specify encoding explicitly
    )
    quantex_subjects_df['birthday'] = pd.to_datetime(quantex_subjects_df['birthday'])
    quantex_subjects_df.to_sql('Subjects', conn, if_exists='replace', index=False)    
    # Convert birthday column to DATE format
    cursor.execute('''
        UPDATE Subjects
        SET birthday = DATE(birthday)
    ''')
    
    conn.commit()
    conn.close()
    logging.info(f"Detection database created at {db_path}")
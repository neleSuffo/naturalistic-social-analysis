import sqlite3
import logging
import pandas as pd
from pathlib import Path
from constants import DetectionPaths, DetectionPipeline, BasePaths

logging.basicConfig(level=logging.INFO)

def get_age_group(age: float) -> int:
    """
    Determines age group based on age.
    
    Parameters:
    ----------
    age : float
        Age in years
        
    Returns:
    -------
    int
        Age group (3, 4, or 5)
    """
    if age < 4:
        return 3
    elif age < 5:
        return 4
    else:
        return 5
    
def create_age_group_file(subjects_df: pd.DataFrame, video_paths: list) -> None:
    """
    Creates age_group.csv file containing video names and corresponding ages at recording.
    
    Parameters:
    ----------
    subjects_df : pd.DataFrame
        DataFrame containing subject information including birthdays
    video_paths : list
        List of video paths to process
    """
    # Initialize empty lists to store data
    video_data = []
    
    # First, log the DataFrame columns to verify column names
    logging.info(f"Available columns in subjects_df: {subjects_df.columns.tolist()}")
    
    for video_path in video_paths:
        # Extract video name without extension
        video_name = Path(video_path).stem
        
        # Extract child ID and recording date using regex
        import re
        id_match = re.search(r'(?<=id)\d{6}', video_name)
        date_match = re.search(r'(\d{4})_(\d{2})_(\d{2})', video_name)
        
        if id_match and date_match:
            try:
                # Get child ID
                child_id = int(id_match.group(0))
                
                # Convert recording date string to datetime
                year, month, day = map(int, date_match.groups())
                recording_date = pd.to_datetime(f"{day}.{month}.{year}", format="%d.%m.%Y")
                
                # Check if child_id exists in the DataFrame
                if child_id not in subjects_df['id'].values:  # Changed from 'id' to 'child_id'
                    logging.warning(f"Child ID {child_id} not found in subjects data for video {video_name}")
                    continue
                
                # Get birthday for this child ID
                child_birthday = subjects_df.loc[subjects_df['id'] == child_id, 'birthday'].iloc[0]  # Changed from 'id' to 'child_id'
                
                # Calculate age at recording
                age_at_recording = (recording_date - child_birthday).days / 365.25
                
                # Determine age group
                age_group = get_age_group(age_at_recording)
                
                video_data.append({
                    'video_name': video_name,
                    'child_id': child_id,
                    'recording_date': recording_date.strftime('%d.%m.%Y'),
                    'age_at_recording': round(age_at_recording, 2),
                    'age_group': age_group
                })
                
            except Exception as e:
                logging.error(f"Error processing video {video_name}: {str(e)}")
                continue
    
    if not video_data:
        logging.warning("No video data was processed successfully!")
        return
        
    # Create DataFrame and save to CSV
    age_group_df = pd.DataFrame(video_data)
    output_path = BasePaths.data_dir / 'age_group.csv'
    age_group_df.to_csv(output_path, index=False)
    logging.info(f"Age group file created at {output_path} with {len(age_group_df)} entries")
    
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

    # Get list of video paths from your video directory
    video_paths = list(Path(DetectionPaths.quantex_videos_input_dir).rglob("*.MP4"))
    logging.info(f"Found {len(video_paths)} video files.")
       
    # Load subjects data from CSV with header row
    quantex_subjects_df = pd.read_csv(
        DetectionPipeline.quantex_subjects,
        header=0,  # Use first row as column names
        sep=';',   # Specify semicolon as separator
        encoding='utf-8',  # Specify encoding explicitly
        parse_dates=['birthday'],  # Specify which columns contain dates
        dayfirst=True  # Specify that dates are in DD.MM.YYYY format
    )
    # Create age group file
    create_age_group_file(quantex_subjects_df, video_paths)
    quantex_subjects_df.to_sql('Subjects', conn, if_exists='replace', index=False)
    # Convert birthday column to DATE format
    cursor.execute('''
        UPDATE Subjects
        SET birthday = DATE(birthday)
    ''')
    
    conn.commit()
    conn.close()
    logging.info(f"Detection database created at {db_path}")
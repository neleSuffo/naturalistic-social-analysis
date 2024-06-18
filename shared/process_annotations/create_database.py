import sqlite3
import json
from projects.social_interactions.src.common.constants import DetectionPaths


def create_db_annotations() -> None:
    """
    This function creates a SQLite database from the annotations JSON file.
    """
    # Create a new SQLite database (or connect to an existing one)
    conn = sqlite3.connect(DetectionPaths.annotations_db_path)
    cursor = conn.cursor()
    
    # Drop tables if they exist
    cursor.execute('DROP TABLE IF EXISTS annotations')
    cursor.execute('DROP TABLE IF EXISTS images')
    cursor.execute('DROP TABLE IF EXISTS videos')


    # Create tables for annotations, images, videos and video file names and ids
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY,
            image_id INTEGER,
            video_id TEXT,
            category_id INTEGER,
            bbox TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            video_id TEXT,
            frame_id INTEGER,
            file_name TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id TEXT PRIMARY KEY,
            file_name TEXT
        )
    ''')

    # Load annotations and insert them into the database
    with open(DetectionPaths.annotations_json_path, 'r') as annotation_file:
        data = json.load(annotation_file)

    # Insert annotations
    for annotation in data['annotations']:
        bbox = json.dumps(annotation['bbox'])  # Convert bbox to JSON string
        cursor.execute('''
            INSERT INTO annotations (id, image_id, video_id, category_id, bbox)
            VALUES (?, ?, ?, ?, ?)
        ''', (annotation['id'], annotation['image_id'], annotation['video_id'], annotation['category_id'], bbox))

    # Insert images
    for image in data['images']:
        cursor.execute('''
            INSERT INTO images (id, video_id, frame_id, file_name)
            VALUES (?, ?, ?, ?)
        ''', (image['id'], image['video_id'], image['frame_id'], image['file_name']))

    # Insert videos
    for video in data['videos']:
        cursor.execute('''
            INSERT INTO videos (id, file_name)
            VALUES (?, ?)
        ''', (video['id'], video['file_name']))

    # Commit and close the database connection
    conn.commit()
    conn.close()


def create_db_table_video_name_id_mapping(
    task_file_id_dict: dict
) -> None:
    """
    This function creates a SQLite database from the annotations xml file.
    
    Parameters
    ----------
    task_file_id_dict : dict
        a dictionary with the task name as key and a dictionary as value
        key: file name, value: file id
    """
    # Create a new SQLite database (or connect to an existing one)
    conn = sqlite3.connect(DetectionPaths.annotations_db_path)
    cursor = conn.cursor()
    
    # Drop tables if they exist
    cursor.execute('DROP TABLE IF EXISTS video_name_id_mapping')
    
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS video_name_id_mapping (
                video_file_name TEXT PRIMARY KEY,
                video_file_id TEXT
            )
        ''')

    # Insert video file name and id mapping
    for file_name, file_id in task_file_id_dict.items():
        cursor.execute('''
            INSERT INTO video_name_id_mapping (video_file_name, video_file_id)
            VALUES (?, ?)
        ''', (file_name, file_id))

    # Commit and close the database connection
    conn.commit()
    conn.close()

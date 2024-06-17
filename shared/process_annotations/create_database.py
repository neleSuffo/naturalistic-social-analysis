import sqlite3
import json
from projects.social_interactions.src.common.constants import DetectionPaths


def create_database():
    """
    This function creates a SQLite database from the annotations JSON file.
    """
    # Create a new SQLite database (or connect to an existing one)
    conn = sqlite3.connect(DetectionPaths.annotations_db_path)
    cursor = conn.cursor()


    # Create tables for annotations, images, and videos
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

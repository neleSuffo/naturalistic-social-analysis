import sqlite3
import json
import cv2
import logging
from pathlib import Path
from src.projects.social_interactions.common.constants import (
    DetectionPaths,
    DetectionParameters,
    VideoParameters,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_db_annotations() -> None:
    """
    This function creates a SQLite database from the annotations JSON file.
    """
    logging.info("Starting to create database annotations.")

    # Create the directory if it does not exist
    Path(DetectionPaths.annotations_db_path).parent.mkdir(parents=True, exist_ok=True)

    # Create a new SQLite database (or connect to an existing one)
    conn = sqlite3.connect(DetectionPaths.annotations_db_path)
    cursor = conn.cursor()

    # Drop tables if they exist
    cursor.execute("DROP TABLE IF EXISTS annotations")
    cursor.execute("DROP TABLE IF EXISTS images")
    cursor.execute("DROP TABLE IF EXISTS videos")

    # Create tables for annotations, images, videos and video file names and ids
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY,
            image_id INTEGER,
            video_id TEXT,
            category_id INTEGER,
            bbox TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            video_id TEXT,
            frame_id INTEGER,
            file_name TEXT,
            file_name_seconds TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id TEXT PRIMARY KEY,
            file_name TEXT,
            frame_width INTEGER,
            frame_height INTEGER
        )
    """)

    # Load annotations and insert them into the database
    with open(DetectionPaths.annotations_json_path, "r") as annotation_file:
        data = json.load(annotation_file)

    # Insert annotations
    for annotation in data["annotations"]:
        bbox = json.dumps(annotation["bbox"])  # Convert bbox to JSON string
        cursor.execute(
            """
            INSERT INTO annotations (id, image_id, video_id, category_id, bbox)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                annotation["id"],
                annotation["image_id"],
                annotation["video_id"],
                annotation["category_id"],
                bbox,
            ),
        )

    # Insert images
    for image in data["images"]:
        # Get the frame number in seconds
        seconds = int(image["frame_id"]) / DetectionParameters.frame_step
        file_name_base, _ = image["file_name"].rsplit(
            "_", 1
        )  # Split on the last underscore
        file_name_seconds = f"{file_name_base}_{int(seconds):06d}"
        cursor.execute(
            """
            INSERT INTO images (id, video_id, frame_id, file_name, file_name_seconds)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                image["id"],
                image["video_id"],
                image["frame_id"],
                image["file_name"],
                file_name_seconds,
            ),
        )

    def get_frame_width_height(video_file_name: str) -> tuple:
        """
        This function gets the frame width and height of a video file.

        Parameters
        ----------
        video_file_name: str
            the video file name

        Returns
        -------
        tuple
            the frame width and height
        """
        video_file_path = DetectionPaths.videos_input / video_file_name
        # Return default frame width and height if the video file does not exist
        if not video_file_path.exists():
            return (VideoParameters.frame_width, VideoParameters.frame_height)
        cap = cv2.VideoCapture(str(video_file_path))
        if not cap.isOpened():
            return (VideoParameters.frame_width, VideoParameters.frame_height)
        # Get the frame width and height
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return frame_width, frame_height

    # Insert videos
    for video in data["videos"]:
        # Get the frame width and height of the video
        frame_width, frame_height = get_frame_width_height(video["file_name"])

        cursor.execute(
            """
            INSERT INTO videos (id, file_name, frame_width, frame_height)
            VALUES (?, ?, ?, ?)
        """,
            (video["id"], video["file_name"], frame_width, frame_height),
        )

    # Commit and close the database connection
    conn.commit()
    logging.info("Database setup complete. Closing connection.")
    conn.close()


def create_db_table_video_name_id_mapping(task_file_id_dict: dict) -> None:
    """
    This function creates a SQLite database from the annotations xml file.

    Parameters
    ----------
    task_file_id_dict : dict
        a dictionary with the task name as key and a dictionary as value
        key: file name, value: file id
    """
    # Create the directory if it does not exist
    Path(DetectionPaths.annotations_db_path).parent.mkdir(parents=True, exist_ok=True)

    # Create a new SQLite database (or connect to an existing one)
    conn = sqlite3.connect(DetectionPaths.annotations_db_path)
    cursor = conn.cursor()

    # Drop tables if they exist
    cursor.execute("DROP TABLE IF EXISTS video_name_id_mapping")

    cursor.execute("""
            CREATE TABLE IF NOT EXISTS video_name_id_mapping (
                video_file_name TEXT PRIMARY KEY,
                video_file_id TEXT
            )
        """)

    # Insert video file name and id mapping
    for file_name, file_id in task_file_id_dict.items():
        cursor.execute(
            """
            INSERT INTO video_name_id_mapping (video_file_name, video_file_id)
            VALUES (?, ?)
        """,
            (file_name, file_id),
        )

    # Commit and close the database connection
    conn.commit()
    conn.close()

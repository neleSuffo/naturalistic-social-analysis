import sqlite3
import json
import cv2
import logging
import re
from pathlib import Path
import xml.etree.ElementTree as ET
from src.projects.social_interactions.common.constants import (
    DetectionPaths,
    VideoParameters,
    LabelToCategoryMapping,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def process_filename(
    filename: str
) -> str:
    """
    This function processes the filename of an image.
    The input filename is structured like "quantex_at_home_id255237_2022_05_08_02_42630.jpg".
    The output filename is structured like "quantex_at_home_id255237_2022_05_08_02_042630".


    Parameters
    ----------
    filename : str
        the filename of the image

    Returns
    -------
    str
        the processed filename
    """
    # Remove the .jpg extension
    base_name = filename.replace('.jpg', '')
    
    # Extract the last number using regex
    match = re.search(r'(\d+)$', base_name)
    
    if match:
        last_number = match.group(1)
        main_part = base_name[:match.start()]
        
        # Zero-pad the last number to six digits
        last_number_padded = last_number.zfill(6)
        
        # Reconstruct the filename
        new_filename = f"{main_part}{last_number_padded}"
    else:
        # If there's no match, return the base name as is (though it should always match)
        new_filename = base_name
    
    return new_filename


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
    #id INTEGER PRIMARY KEY,
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            image_id INTEGER,
            video_id INTEGER,
            category_id INTEGER,
            bbox TEXT
        )
    """)
    
    #id INTEGER PRIMARY KEY,
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            video_id INTEGER,
            frame_id INTEGER,
            file_name TEXT,
            file_name_storage TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY,
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
        bbox = json.dumps(annotation["bbox"])  # Convert bbox to JSON strin
        #INSERT INTO annotations (id, image_id, video_id, category_id, bbox)
        cursor.execute(
            """
            INSERT INTO annotations (image_id, video_id, category_id, bbox)
            VALUES (?, ?, ?, ?)
        """,
            (
                #annotation["id"],
                annotation["image_id"],
                annotation["video_id"],
                annotation["category_id"],
                bbox,
            ),
        )

    # Insert images
    for image in data["images"]:
    #INSERT INTO images (id, video_id, frame_id, file_name, file_name_storage)

        cursor.execute(
            """
            INSERT INTO images (video_id, frame_id, file_name, file_name_storage)

            VALUES (?, ?, ?, ?)
        """,
            (
                #image["id"],
                image["video_id"],
                image["frame_id"],
                image["file_name"],
                process_filename(image["file_name"])
,
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

def correct_erronous_videos_in_db() -> None:
    """
    This function deletes erroneous videos in the database.
    """    
    # Delete annotations and images for videos with video_id > 100 
    # Connect to the SQLite database
    conn = sqlite3.connect(DetectionPaths.annotations_db_path)
    cursor = conn.cursor()
    # The condition for deletion
    video_id_threshold = 100
    cursor.execute("DELETE FROM annotations WHERE video_id > ?", (video_id_threshold,))
    cursor.execute("DELETE FROM images WHERE video_id > ?", (video_id_threshold,))
    conn.commit()
    logging.info("Finished deleting erroneous videos in the database.")
    conn.close()


def add_annotations_to_db(
    xml_path: Path
) -> None:
    """ 
    This function adds annotations from an XML file to the database.
    
    Parameters
    ----------
    xml_path : Path
        the path to the XML file
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(DetectionPaths.annotations_db_path)
    cursor = conn.cursor()

    # Load the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Navigate to the <task> element
    task_element = root.find('meta/task')

    # Extract <id> and <name> values
    task_id = task_element.find('id').text
    task_name = task_element.find('name').text

    # Iterate over all 'track' elements
    for track in root.iter("track"):
        track_label = track.get("label")

        # Map the label to its corresponding label id using the dictionary
        # returns -1 if the label is not in the dictionary
        track_label_id = LabelToCategoryMapping.label_dict.get(track_label, -1)

        # Iterate over each 'box' element within the 'track'
        for box in track.iter("box"):
            row = box.attrib
            frame_padded = f'{int(row["frame"]):06}'
            bbox_json = json.dumps([float(row["xtl"]), float(row["ytl"]), float(row["xbr"]), float(row["ybr"])])

            # Insert the annotation into the database
            cursor.execute(
                """
                INSERT INTO annotations (image_id, video_id, category_id, bbox)
                VALUES (?, ?, ?, ?)
            """,
                (
                    row["frame"], # image_id
                    task_id, # video_id
                    track_label_id, # category_id
                    bbox_json, # bbox coordinates
                ),
            )

            cursor.execute(
                """
                INSERT INTO images (video_id, frame_id, file_name, file_name_storage)

                VALUES (?, ?, ?, ?)
            """,
                (
                    task_id, # video_id
                    row["frame"], # frame_id
                    f'{task_name}_{frame_padded}', # file_name
                    f'{task_name}_{frame_padded}', # file_name
                ),
            )
    conn.commit()
    logging.info(f'Database commit for file {xml_path} successful')
    conn.close()


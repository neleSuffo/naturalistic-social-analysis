import sqlite3
import json
import cv2
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from constants import DetectionPaths
from prepare_data.process_annotations.utils import (
    get_task_ids_names_lengths,
    generate_cum_sum_frame_count,
    get_video_id_from_name_db,
)
from config import VideoConfig, LabelToCategoryMapping


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


def write_xml_to_database() -> None:
    """
    This function creates a SQLite database from the annotations XML file.
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
    cursor.execute("DROP TABLE IF EXISTS categories")

    # Create tables for annotations, images, and videos
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            image_id INTEGER,
            video_id INTEGER,
            category_id INTEGER,
            bbox TEXT,
            outside INTEGER,
            person_visibility INTEGER,
            person_ID INTEGER,
            person_age TEXT,
            person_gender TEXT,
            object_interaction BOOLEAN
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            video_id INTEGER,
            frame_id INTEGER,
            file_name TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY,
            file_name TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY,
            category TEXT,
            supercategory TEXT
        )
    """)

    # Load and parse the XML annotations
    tree = ET.parse(DetectionPaths.annotations_xml_path)
    root = tree.getroot()

    # Extract task ids, the name and length of the task
    task_details = get_task_ids_names_lengths(root)
    
     # Get the task ids from the XML file (the ones that have tracks)
    task_ids_in_tracks = {track.get("task_id") for track in root.iter("track")}
    
    # Remove keys from task_details that are not present in task_ids_in_tracks
    task_details_reduced = {task_id: details for task_id, details in task_details.items() if task_id in task_ids_in_tracks}
    
    # Generate the frame correction dictionary 
    # (key: task id, value: (task name, task length, frame correction))
    frame_correction_dict = generate_cum_sum_frame_count(task_details)
    
    # Initialize the frame correction value
    frame_correction = 0
    
    # Initialize an empty set for added categories and videos
    added_categories = set()
    added_videos = set()
    added_images = set()
    
    # Extract and insert annotations
    for track in root.iter("track"):
        task_id = track.get("task_id")
        track_label = track.get("label")
        # Map the label to its corresponding label id using the dictionary
        # returns -1 if the label is not in the dictionary
        track_label_id = LabelToCategoryMapping.label_dict.get(track_label, LabelToCategoryMapping.unknown_label_id)
        # Map the label to its corresponding supercategory using the dictionary
        supercategory = LabelToCategoryMapping.supercategory_dict.get(
            track_label_id, LabelToCategoryMapping.unknown_supercategory
        )  # returns "unknown" if the label is not in the dictionary

        # Get the frame correction value
        if task_id is not None:
            frame_correction = frame_correction_dict[task_id][2]
        # Get the task name from the task_details dictionary
        task_name = task_details.get(task_id, next(iter(task_details.values())))[0]
        # Get the video_id from the database using the task_name
        task_id = get_video_id_from_name_db(task_name, cursor)
        
        # Add video details if not already added
        if task_id not in added_videos:
        # Add video details if not already added
            cursor.execute(
            """
            INSERT INTO videos (id, file_name)
            VALUES (?, ?)
            """,
                (
                task_id,
                f"{task_name}.mp4",
                ),
            )
            added_videos.add(task_id)
        
        # Add category details if not already added
        if track_label_id not in added_categories:
            cursor.execute(
            """
            INSERT INTO categories (id, category, supercategory)
            VALUES (?, ?, ?)
            """,
                (
                track_label_id,
                track_label,
                supercategory,
                ),
            )
            added_categories.add(track_label_id)

        # Iterate over each 'box' element within the 'track'
        for box in track.iter("box"):
            row = box.attrib
            frame_id = int(row["frame"]) - frame_correction
            outside = int(row["outside"]) # 0 if the object is inside the frame, 1 if it is outside
            frame_id_padded = f'{frame_id:06}'
            bbox_json = json.dumps([float(row["xtl"]), float(row["ytl"]), float(row["xbr"]), float(row["ybr"])])

            # Extract other attributes
            person_visibility = box.find(".//attribute[@name='Visibility']")
            person_visibility_value = int(person_visibility.text) if person_visibility is not None else None

            person_id = box.find(".//attribute[@name='ID']")
            person_id_value = int(person_id.text) if person_id is not None else None

            person_age = box.find(".//attribute[@name='Age']")
            person_age_value = person_age.text if person_age is not None else None

            person_gender = box.find(".//attribute[@name='Gender']")
            person_gender_value = person_gender.text if person_gender is not None else None

            object_interaction = box.find(".//attribute[@name='Interaction']")
            object_interaction_value = object_interaction.text if object_interaction is not None else "No"

            # Insert the annotation into the database
            cursor.execute(
            """
                INSERT INTO annotations (image_id, video_id, category_id, bbox, outside, person_visibility, person_ID, person_age, person_gender, object_interaction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                frame_id,
                task_id,
                track_label_id,
                bbox_json,
                outside,
                person_visibility_value,
                person_id_value,
                person_age_value,
                person_gender_value,
                object_interaction_value,
                ),
            )
            # Add image details if not already added
            image_name = f"{task_name}_{frame_id_padded}.jpg"
            if image_name not in added_images:
                cursor.execute(
                """
                    INSERT INTO images (video_id, frame_id, file_name)
                    VALUES (?, ?, ?)
                """,
                    (
                    task_id,
                    frame_id,
                    image_name,
                    ),
                ) 
                added_images.add(image_name)


    # Commit and close the database connection
    conn.commit()
    logging.info("Database setup complete. Closing connection.")
    conn.close()


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
    video_file_path = DetectionPaths.videos_input_dir / video_file_name
    # Return default frame width and height if the video file does not exist
    if not video_file_path.exists():
        return (VideoConfig.frame_width, VideoConfig.frame_height)
    cap = cv2.VideoCapture(str(video_file_path))
    if not cap.isOpened():
        return (VideoConfig.frame_width, VideoConfig.frame_height)
    # Get the frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return frame_width, frame_height


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
    DetectionPaths.annotations_db_path.parent.mkdir(parents=True, exist_ok=True)

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
    It also adds the correct annotations to the database.
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
    
    # Add the correct annotations to the database
    for file_name in DetectionPaths.annotations_individual_dir.iterdir():
        add_annotations_to_db(cursor, conn, file_name)
    conn.close()


def add_annotations_to_db(
    cursor: sqlite3.Cursor,
    conn: sqlite3.Connection,
    xml_path: Path
) -> None:
    """ 
    This function adds annotations from an XML file to the database.
    
    Parameters
    ----------
    cursor : sqlite3.Cursor
        the cursor object
    conn : sqlite3.Connection
        the connection object
    xml_path : Path
        the path to the XML file
    """
    # Initialize an empty set for added categories and videos
    added_images = set()

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
        # Map the label to its corresponding supercategory using the dictionary
        supercategory = LabelToCategoryMapping.supercategory_dict.get(
            track_label_id, LabelToCategoryMapping.unknown_supercategory
        )  # returns "unknown" if the label is not in the dictionary

        # Iterate over each 'box' element within the 'track'
        for box in track.iter("box"):
            row = box.attrib
            outside = int(row["outside"]) # 0 if the object is inside the frame, 1 if it is outside
            frame_id_padded = f'{int(row["frame"]):06}'
            bbox_json = json.dumps([float(row["xtl"]), float(row["ytl"]), float(row["xbr"]), float(row["ybr"])])
            
            # Extract other attributes
            person_visibility = box.find(".//attribute[@name='Visibility']")
            person_visibility_value = int(person_visibility.text) if person_visibility is not None else None

            person_id = box.find(".//attribute[@name='ID']")
            person_id_value = int(person_id.text) if person_id is not None else None

            person_age = box.find(".//attribute[@name='Age']")
            person_age_value = person_age.text if person_age is not None else None

            person_gender = box.find(".//attribute[@name='Gender']")
            person_gender_value = person_gender.text if person_gender is not None else None

            object_interaction = box.find(".//attribute[@name='Interaction']")
            object_interaction_value = object_interaction.text if object_interaction is not None else "No"

            # Insert the annotation into the database
            cursor.execute(
                """
                INSERT INTO annotations (image_id, video_id, category_id, bbox, outside, person_visibility, person_ID, person_age, person_gender, object_interaction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    row["frame"], # image_id
                    task_id, # video_id
                    track_label_id, # category_id
                    bbox_json, # bbox coordinates
                    outside,
                    person_visibility_value,
                    person_id_value,
                    person_age_value,
                    person_gender_value,
                    object_interaction_value,
                ),
            )
            # Add image details if not already added
            image_name = f"{task_name}_{frame_id_padded}.jpg"
            if image_name not in added_images:
                cursor.execute(
                """
                    INSERT INTO images (video_id, frame_id, file_name)
                    VALUES (?, ?, ?)
                """,
                    (
                    task_id, # video_id
                    row["frame"], # frame_id
                    image_name,
                    ),
                ) 
                added_images.add(image_name)
    conn.commit()
    logging.info(f'Database commit for file {xml_path} successful')


def create_child_class_in_db():
    """
    This function creates a new class "child_body
    """
    conn = sqlite3.connect(DetectionPaths.annotations_db_path)
    cursor = conn.cursor()
    # Update the category_id for child body parts
    query_1 = """
    UPDATE annotations
    SET category_id = 11
    WHERE category_id = 1 AND person_ID = 1;
    """
    # Add new category to the categories table
    query_2 = """
    INSERT INTO categories (id, category, supercategory)
    VALUES (11, 'child_body_part', 'person');
    """
    # Execute the query
    cursor.execute(query_1)
    logging.info("Successfully updated category_id for child body parts.")
    cursor.execute(query_2)
    logging.info("Successfully added new category 'child_body_part' to the categories table.")

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

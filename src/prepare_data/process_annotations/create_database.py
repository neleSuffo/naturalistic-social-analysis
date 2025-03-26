import sqlite3
import json
import cv2
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from constants import DetectionPaths
from prepare_data.process_annotations.anno_utils import (
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
    Path(DetectionPaths.quantex_annotations_db_path).parent.mkdir(parents=True, exist_ok=True)

    added_images = set()  # Missing set initialization
    added_videos = set()
    added_categories = set()
    
    with sqlite3.connect(DetectionPaths.quantex_annotations_db_path) as conn:
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
                gaze_directed_at_child TEXT,
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

        # Commit and close the database connection
        conn.commit()
        for file_name in DetectionPaths.quantex_annotations_dir.iterdir():
            if file_name.suffix == '.xml':
                add_annotations_to_db(cursor, conn, file_name, added_images, added_videos, added_categories)
                
        logging.info("Database setup complete.")

def get_frame_width_height(video_file_name: str) -> tuple:
    try:
        video_file_path = DetectionPaths.quantex_videos_input_dir / video_file_name
        if not video_file_path.exists():
            logging.warning(f"Video file {video_file_path} not found, using default dimensions")
            return (VideoConfig.frame_width, VideoConfig.frame_height)
            
        cap = cv2.VideoCapture(str(video_file_path))
        if not cap.isOpened():
            logging.warning(f"Could not open video file {video_file_path}, using default dimensions")
            return (VideoConfig.frame_width, VideoConfig.frame_height)
            
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return frame_width, frame_height
        
    except Exception as e:
        logging.error(f"Error getting frame dimensions: {e}")
        return (VideoConfig.frame_width, VideoConfig.frame_height)

def add_annotations_to_db(
    cursor: sqlite3.Cursor,
    conn: sqlite3.Connection,
    xml_path: Path,
    added_images: set,
    added_videos: set,
    added_categories: set
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
    # Load the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Navigate to the <task> element
    task_element = root.find('meta/task')

    # Extract <id> and <name> values
    task_id = task_element.find('id').text
    task_name = task_element.find('name').text
    
    # Add video details if not already added
    if task_id not in added_videos:
        cursor.execute(
            "INSERT INTO videos (id, file_name) VALUES (?, ?)",
            (task_id, f"{task_name}.mp4")
        )
        added_videos.add(task_id)

    # Iterate over all 'track' elements
    for track in root.iter("track"):
        track_label = track.get("label")

        # Map the label to its corresponding label id using the dictionary
        # returns -1 if the label is not in the dictionary
        track_label_id = LabelToCategoryMapping.label_to_id_mapping.get(track_label, -1)
        # Map the label to its corresponding supercategory using the dictionary
        supercategory = LabelToCategoryMapping.id_to_supercategory_mapping.get(
            track_label_id, LabelToCategoryMapping.unknown_supercategory
        )  # returns "unknown" if the label is not in the dictionary

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
            person_gender_value = person_gender.text if person_gender is not None else None#
            
            gaze_directed_at_child = box.find(".//attribute[@name='Gaze Directed at Child']")
            gaze_directed_at_child_value = gaze_directed_at_child.text if gaze_directed_at_child is not None else None

            object_interaction = box.find(".//attribute[@name='Interaction']")
            object_interaction_value = object_interaction.text if object_interaction is not None else "No"

            # Insert the annotation into the database
            cursor.execute(
                """
                INSERT INTO annotations (image_id, video_id, category_id, bbox, outside, person_visibility, person_ID, person_age, person_gender, gaze_directed_at_child, object_interaction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    gaze_directed_at_child_value,
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
    This function creates a new class "child_body part" in the database.
    """
    conn = sqlite3.connect(DetectionPaths.quantex_annotations_db_path)
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

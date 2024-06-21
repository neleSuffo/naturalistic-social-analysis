import torch
import sys
import cv2
import sqlite3
from projects.social_interactions.src.common.constants import YoloParameters, DetectionPaths

sys.path.append('/Users/nelesuffo/projects/leuphana-IPE/yolov5/')

    
def load_yolo_model() -> torch.nn.Module:
    """ 
    Load the YOLOv5 model for person detection.
            
    Returns
    -------
    torch.nn.Module
        The YOLOv5 model.
    """
    weights_path = YoloParameters.pretrained_weights_path
    try:
        # Always download the model from online
        print("Downloading YOLOv5 model from online...")
        yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

        # Save the model weights, overwriting the existing file if it exists
        torch.save(yolo_model.state_dict(), weights_path)

        print("Successfully loaded and saved the YOLOv5 model.")
        return yolo_model
    except Exception as e:
        print(f"Error occurred while loading the YOLOv5 model: {e}")
        raise


def fetch_all_annotations(
    db_path: str = DetectionPaths.annotations_db_path,
    category_ids: list[int] = [1, 2],
) -> list:
    """
    This function fetches all annotations from the database.

    Parameters
    ----------
    db_path : str
        the path to the database
    category_ids : list[int], optional
        the list of category ids, by default [1, 2]
        (face and reflection)

    Returns
    -------
    list
        the list of annotations
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Generate a string of ? placeholders that matches the length of category_ids
    placeholders = ', '.join('?' for _ in category_ids)
    
    # Fetch all annotations
    # filter only annotations with category_id 1 or 2 (person or reflection)
    cursor.execute(f'''
    SELECT DISTINCT a.id, a.image_id, a.video_id, a.category_id, a.bbox, i.file_name_seconds as image_file_name, v.file_name as video_file_name, v.frame_width, v.frame_height
    FROM annotations a 
    JOIN images i ON a.image_id = i.frame_id AND a.video_id = i.video_id
    JOIN videos v ON a.video_id = v.id
    WHERE a.category_id IN ({placeholders})
    ORDER BY a.video_id, a.image_id
    ''', category_ids)
    
    annotations = cursor.fetchall()
    conn.close()
    
    return annotations

def get_frame_width_height(
    video_file_path: str
) -> tuple:
    """
    This function gets the frame width and height of a video file.

    Parameters
    ----------
    video_file_path : str
        _description_

    Returns
    -------
    tuple
        _description_
    """

# Open the video file
    cap = cv2.VideoCapture('path_to_your_video_file')

    # Get the frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Print the frame size
    print('Frame size:', frame_width, 'x', frame_height)

    # Always release the VideoCapture object
    cap.release()
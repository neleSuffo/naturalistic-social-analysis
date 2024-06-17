import torch
import logging
import os
import shutil
import sqlite3

def load_yolov5_model() -> torch.nn.Module:
    """
    This function loads the person detection model.

    Returns
    -------
    torch.nn.Module
        the person detection model
    """
    try:
        # Load the YOLOv5 model for person detection
        yolov5_model = torch.hub.load("ultralytics/yolov5", "yolov5s")
        # Move the model to the desired location
        if os.path.exists("yolov5s.pt"):
            shutil.move(
                "yolov5s.pt",
                "/Users/nelesuffo/projects/leuphana-IPE/pretrained_models/yolov5s.pt",
            )
        logging.info("Successfully downloaded and loaded the person detection model.")
        return yolov5_model
    except Exception as e:
        logging.error(f"Error occurred while loading the person detection model: {e}")
        raise


def fetch_all_annotations(db_path: str) -> list:
    """
    This function fetches all annotations from the database.

    Parameters
    ----------
    db_path : str
        the path to the database

    Returns
    -------
    list
        the list of annotations
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT a.id, a.image_id, a.video_id, a.bbox, i.file_name as image_file_name, v.file_name as video_file_name 
        FROM annotations a 
        JOIN images i ON a.image_id = i.id 
        JOIN videos v ON a.video_id = v.id
    ''')
    
    annotations = cursor.fetchall()
    conn.close()
    
    return annotations
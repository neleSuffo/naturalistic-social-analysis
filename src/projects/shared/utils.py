import torch
import cv2
import sqlite3
from facenet_pytorch import MTCNN
from typing import List, Optional
from src.projects.social_interactions.common.constants import (
    YoloParameters,
    DetectionPaths,
)


def load_yolov5_model(
    model_name: str = "yolov5s",
    local_path: str = YoloParameters.pretrained_weights_path,
) -> torch.nn.Module:
    """
    This function loads the YOLOv5 model from the local path if it exists,
    otherwise it downloads the model from the internet and saves it locally.

    Parameters
    ----------
    model_name : str, optional
        the name of the model, by default 'yolov5s'
    local_path : str, optional
        the local path to save the model, by default 'yolov5s.pt'

    Returns
    -------
    torch.nn.Module
        the YOLOv5 model

    """
    print(f"Downloading model: {model_name}")
    model = torch.hub.load("ultralytics/yolov5", model_name, pretrained=True)
    # Save the model locally for future use
    torch.save(model.state_dict(), local_path)
    return model


def load_mtcnn_model() -> MTCNN:
    """
    This function loads the MTCNN model.

    Returns
    -------
    MTCNN
        the MTCNN model

    """
    # Load the MTCNN model
    mtcnn = MTCNN(keep_all=True)
    return mtcnn


def fetch_all_annotations(
    db_path: str = DetectionPaths.annotations_db_path,
    category_ids: Optional[List[int]] = None,
) -> List[tuple]:
    """
    This function fetches all annotations from the database (excluding the -1 category id)
    Categories with id -1 are labeled as "ignore" and are not included in the annotations.

    Parameters
    ----------
    db_path : str
        the path to the database file,
        by default DetectionPaths.annotations_db_path
    category_ids : list, optional
        the list of category ids to filter the annotations,
        by default None


    Returns
    -------
    list of tuple
        the list of annotations
        (image_id, video_id, category_id, bbox, image_file_name, video_file_name)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if category_ids:
        # Generate a string of ? placeholders that matches the length of category_ids
        placeholders = ", ".join("?" for _ in category_ids)

    if category_ids:
        # Generate a string of ? placeholders that matches the length of category_ids
        placeholders = ", ".join("?" for _ in category_ids)
        query = f"""
        SELECT DISTINCT a.image_id, a.video_id, a.category_id, a.bbox, i.file_name, v.file_name as video_file_name
        FROM annotations a
        JOIN images i ON a.image_id = i.frame_id AND a.video_id = i.video_id
        JOIN videos v ON a.video_id = v.id
        WHERE a.category_id IN ({placeholders}) AND a.category_id != -1
        ORDER BY a.video_id, a.image_id
        """
        cursor.execute(query, category_ids)
    else:
        query = """
        SELECT DISTINCT a.image_id, a.video_id, a.category_id, a.bbox, i.file_name, v.file_name as video_file_name
        FROM annotations a
        JOIN images i ON a.image_id = i.frame_id AND a.video_id = i.video_id
        JOIN videos v ON a.video_id = v.id
        WHERE a.category_id != -1
        ORDER BY a.video_id, a.image_id
        """
        cursor.execute(query)

    annotations = cursor.fetchall()
    conn.close()
    return annotations


def get_frame_width_height(video_file_path: str) -> tuple:
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
    cap = cv2.VideoCapture("path_to_your_video_file")

    # Get the frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Print the frame size
    print("Frame size:", frame_width, "x", frame_height)

    # Always release the VideoCapture object
    cap.release()

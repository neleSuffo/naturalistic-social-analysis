import os
import sqlite3
import numpy as np
import random
from typing import Dict, List, Tuple
from pathlib import Path
from src.projects.social_interactions.common.constants import DetectionPaths as DP, TrainParameters as TP, FastReIDParameters as FRP
from PIL import Image


def fetch_all_annotations() -> List[Tuple[int, int, int, str, int, str, str]]:
    """
    This function fetches all annotations from the database, excluding the -1 category id.
    
    Parameters
    ----------
    db_path : str
        the path to the database file
    category_ids : list, optional
        the list of category ids to filter the annotations
    
    Returns
    -------
    list of tuple
        the list of annotations
        (image_id, video_id, category_id, bbox, image_file_name, video_file_name)
    """
    conn = sqlite3.connect(DP.annotations_db_path)
    cursor = conn.cursor()


    query = """
    SELECT DISTINCT a.image_id, a.video_id, a.category_id, a.bbox, a.person_id, i.file_name, v.file_name as video_file_name
    FROM annotations a
    JOIN images i ON a.image_id = i.frame_id AND a.video_id = i.video_id
    JOIN videos v ON a.video_id = v.id
    WHERE a.category_id != -1 AND a.category_id = 1 AND a.outside = 0 AND a.video_id = 129
    ORDER BY a.video_id, a.image_id
    """
    cursor.execute(query)

    annotations = cursor.fetchall()
    conn.close()
    return annotations


def split_videos(annotations: List[Tuple[int, int, int, str, int, str, str]]) -> Tuple[List[Tuple[int, int, int, str, int, str, str]], List[Tuple[int, int, int, str, int, str, str]]]:
    """
    This function splits the videos into training and testing sets
    based on the video_id.
    
    Parameters
    ----------
    annotations : list of tuple
        The list of annotations
    
    Returns
    -------
    tuple of (list, list)
        The training and testing annotations
    """
    split_ratio = TP.train_test_split
    video_annotations: Dict[int, List[Tuple[int, int, int, str, str, str]]] = {}

    # Group annotations by video ID
    for annotation in annotations:
        _, video_id, _, _, _, _ , _ = annotation
        if video_id not in video_annotations:
            video_annotations[video_id] = []
        video_annotations[video_id].append(annotation)

    # Split videos into train and test sets
    video_ids = list(video_annotations.keys())
    random.shuffle(video_ids)  # Shuffle to ensure randomness
    split_point = int(len(video_ids) * split_ratio)

    train_video_ids = set(video_ids[:split_point])
    test_video_ids = set(video_ids[split_point:])

    train_annotations = [ann for video_id in train_video_ids for ann in video_annotations[video_id]]
    test_annotations = [ann for video_id in test_video_ids for ann in video_annotations[video_id]]

    return train_annotations, test_annotations


def save_cropped_images(
    annotations: List[Tuple[int, int, int, str, str, str]],
    save_folder: Path
):
    """
    This function saves the cropped images based on the annotations.
    
    Parameters
    ----------
    annotations : list of tuple
        The list of annotations
    save_folder : Path
    """
    # Get the path to the images
    images_input = DP.images_input
    os.makedirs(save_folder, exist_ok=True)

    for (image_id, video_id, _, bbox, person_id, image_file_name, _) in annotations:
        # Load the image
        image_path = images_input/image_file_name
        with Image.open(image_path) as img:
            # Extract bbox coordinates
            x, y, w, h = map(int, bbox.strip('()').split(','))

            # Crop the image
            cropped_img = img.crop((x, y, x + w, y + h))

            # Format the filename
            person_id = f'{person_id:04d}'
            video_id_str = f'c{video_id:02d}'
            frame_id = f'f{int(image_id):06d}'
            cropped_image_name = f'{person_id}_{video_id_str}_{frame_id}.jpg'
            cropped_image_path = os.path.join(save_folder, cropped_image_name)

            # Save the cropped image
            cropped_img.save(cropped_image_path)

def main():
    # Fetch all annotations
    annotations = fetch_all_annotations()
    # Split videos into train and test sets
    train_annotations, test_annotations = split_videos(annotations)

    # Save the cropped images
    save_cropped_images(train_annotations, FRP.videos_train)
    save_cropped_images(test_annotations, FRP.videos_val)
    
if __name__ == '__main__':
    main()
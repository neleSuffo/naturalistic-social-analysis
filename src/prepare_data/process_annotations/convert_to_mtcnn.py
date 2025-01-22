import json
import logging
from utils import fetch_all_annotations
from constants import MtcnnPaths
from collections import defaultdict
from config import YoloConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_annotations(annotations: list) -> None:
    """
    This function saves the annotations in MTCNN format to text files.

    Parameters
    ----------
    annotations : list
        the list of annotations
        (image_id, video_id, category_id, bbox, image_file_name, video_file_name)
    """
    logging.info("Saving annotations in MTCNN format.")
    output_file_path = MtcnnPaths.labels_file_path
    output_dir = output_file_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    annotations_by_image = defaultdict(list)
    
    for annotation in annotations:
        _, _, _, bbox, image_file_name, _ , gaze = annotation
        bbox = json.loads(bbox)
        xtl, ytl, xbr, ybr = bbox
        width = xbr - xtl
        height = ybr - ytl
        # MTCNN format: xtl, ytl, width, height
        mtcnn_bbox = (xtl, ytl, width, height)
        
        # Map gaze information
        gaze_boolean = 1 if gaze == "Yes" else 0

        annotations_by_image[image_file_name].append((mtcnn_bbox, gaze_boolean))
    
    with open(output_file_path, 'w') as f:
        for image_file_name, bboxes in annotations_by_image.items():
            line = f"{image_file_name} "
            line += " ".join(f"{xtl},{ytl},{width},{height},{gaze_boolean}" for (xtl, ytl, width, height), gaze_boolean in bboxes)
            line += "\n"
            f.write(line)

def main() -> None:
    logging.info("Starting the conversion process for MTCNN.")
    try:
        # Fetch all annotations for category 10 (face)
        annotations = fetch_all_annotations(category_ids = YoloConfig.face_target_class_ids)
        logging.info(f"Fetched {len(annotations)} annotations.")
        save_annotations(annotations)
        logging.info("Successfully saved all annotations.")
    except Exception as e:
        logging.error(f"Failed to fetch annotations or save them: {e}")

if __name__ == "__main__":
    main()

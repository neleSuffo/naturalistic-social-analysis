import json
import logging
from src.projects.shared.utils import fetch_all_annotations
from src.projects.social_interactions.common.constants import MtcnnParameters as Mtcnn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_annotations(annotations: list) -> None:
    """
    This function saves the annotations in MTCNN format to text files.

    Parameters
    ----------
    annotations : list
        the list of annotations
    """
    logging.info("Saving annotations in MTCNN format.")
    output_file_path = Mtcnn.labels_input
    output_dir = output_file_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_file_path, 'a') as f:
        for annotation in annotations:
            _, _, _, bbox, image_file_name, _, _, _ = annotation
            bbox = json.loads(bbox)
            xtl, ytl, xbr, ybr = bbox
            width = xbr - xtl
            height = ybr - ytl
            # MTCNN format: xtl, ytl, width, height
            mtcnn_bbox = (xtl, ytl, width, height)

            line = f"{image_file_name} " + " ".join(map(str, mtcnn_bbox)) + '\n'
            f.write(line)

def main() -> None:
    logging.info("Starting the conversion process for MTCNN.")
    try:
        # Fetch all annotations for category 10 (face)
        annotations = fetch_all_annotations(category_ids=Mtcnn.class_id)
        logging.info(f"Fetched {len(annotations)} annotations.")
        save_annotations(annotations)
        logging.info("Successfully saved all annotations.")
    except Exception as e:
        logging.error(f"Failed to fetch annotations or save them: {e}")

if __name__ == "__main__":
    main()

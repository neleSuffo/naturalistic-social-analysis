import json
import logging
from src.projects.shared.utils import fetch_all_annotations
from src.projects.social_interactions.common.constants import YoloParameters as Yolo

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def convert_bbox(
    size: tuple, 
    bbox: list
) -> tuple:
    """
    This function converts the bounding box coordinates to YOLO format.

    Parameters
    ----------
    size : tuple
        the image size
    bbox : list
        the bounding box coordinates

    Returns
    -------
    tuple
        the bounding box coordinates in YOLO format
    
    Example
    -------
    Input: size=(1280, 720), bbox=[xmin, ymin, xmax, ymax]
    Output: (x_center, y_center, width, height) normalized to [0, 1]
    """
    # Calculate the width and height of the image
    dw = 1. / size[0]
    dh = 1. / size[1]
    # Calculate the center of the bounding box
    x = (bbox[0] + bbox[2]) / 2.0
    y = (bbox[1] + bbox[3]) / 2.0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    # Normalize the coordinates
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def save_annotations(
    annotations: list
) -> None:
    """
    This function saves the annotations in YOLO format to text files.

    Parameters
    ----------
    annotations : list
        the list of annotations
    """
    logging.info("Saving annotations in YOLO format.")
    output_dir = Yolo.labels_input
    output_dir.mkdir(parents=True, exist_ok=True)
    file_contents = {}

    for annotation in annotations:
        _, _, _, category_id, bbox, image_file_name, _, image_width, image_height = annotation
        bbox = json.loads(bbox)
        image_size = (image_width, image_height)
        # YOLO format: category_id x_center y_center width height
        yolo_bbox = convert_bbox(image_size, bbox)
        
        line = f"{category_id} " + " ".join(map(str, yolo_bbox)) + '\n'
        # Append the line to the list of lines for the image file
        if image_file_name not in file_contents:
            file_contents[image_file_name] = []
        file_contents[image_file_name].append(line)

    # Write the lines to text files
    for image_file_name, lines in file_contents.items():
        txt_file = output_dir / (image_file_name + '.txt')
        try:
            with open(txt_file, 'a') as f:
                f.writelines(lines)
        except IOError as e:
            logging.error(f"Failed to write to file {txt_file}: {e}")

def main() -> None:
    logging.info("Starting the conversion process for Yolo.")
    try:
        annotations = fetch_all_annotations(category_ids=Yolo.class_id)
        logging.info(f"Fetched {len(annotations)} annotations.")
        save_annotations(annotations)
        logging.info("Successfully saved all annotations.")
    except Exception as e:
        print(f"Failed to fetch annotations or save them: {e}")

if __name__ == "__main__":
    main()

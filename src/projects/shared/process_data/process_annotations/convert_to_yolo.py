import json
import logging
import cv2
from src.projects.shared.utils import fetch_all_annotations
from src.projects.social_interactions.common.constants import YoloParameters as Yolo, DetectionPaths

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def convert_to_yolo_format(
    image_path, bbox
) -> list:
    """
    This function converts the given annotations to YOLO format.

    Parameters
    ----------
    image_path : str
        the path to the image file
    bbox : list
        the bounding box coordinates

    Returns
    -------
    annotations : list
        the annotations in YOLO format
    """
    # Load the image to get its dimensions
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]

    xtl, ytl, xbr, ybr = bbox
    # Calculate YOLO format coordinates
    x_center = (xtl + xbr) / 2.0 / img_width
    y_center = (ytl + ybr) / 2.0 / img_height
    width = (xbr - xtl) / img_width
    height = (ybr - ytl) / img_height

    return (x_center, y_center, width, height)


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

    #(image_id, video_id, category_id, bbox, image_file_name, video_file_name, frame_width, frame_height)


    for annotation in annotations:
        _, _, category_id, bbox, image_file_name, _, image_width, image_height = annotation
        bbox = json.loads(bbox)
        
        # Consrtuct the path to the image file
        image_file_path = DetectionPaths.images_input / (image_file_name + '.jpg')
        
        # Check if the image file exists
        if not image_file_path.is_file():
            logging.warning(f"Image file {image_file_path} does not exist. Skipping annotation.")
            continue
    
        # YOLO format: category_id x_center y_center width height
        try:
            yolo_bbox = convert_to_yolo_format(image_file_path, bbox)
        except Exception as e:
            logging.error(f"Error converting bbox to YOLO format for {image_file_path}: {e}")
            continue
                
        # Create a line for the text file
        line = f"{category_id} " + " ".join(map(str, yolo_bbox)) + '\n'
        # Append the line to the list of lines for the image file
        if image_file_name not in file_contents:
            file_contents[image_file_name] = []
        file_contents[image_file_name].append(line)

    # Write the lines to text files
    for image_file_name, lines in file_contents.items():
        txt_file = output_dir / (image_file_name + '.txt')
        try:
            with open(txt_file, 'w') as f:
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

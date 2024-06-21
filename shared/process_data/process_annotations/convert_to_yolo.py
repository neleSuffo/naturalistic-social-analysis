import os
import json
from shared.utils import fetch_all_annotations
from projects.social_interactions.src.common.constants import DetectionPaths


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
    annotations: list, 
    output_dir: str
) -> None:
    """
    This function saves the annotations in YOLO format to text files.

    Parameters
    ----------
    annotations : list
        the list of annotations
    output_dir : str
        the output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    for annotation in annotations:
        _, _, _, category_id, bbox, image_file_name, _, image_width, image_height = annotation
        bbox = json.loads(bbox)
        image_size = (image_width, image_height)
        yolo_bbox = convert_bbox(image_size, bbox)
        
        txt_file = os.path.join(output_dir, image_file_name + '.txt')
        try:
            with open(txt_file, 'a') as f:
                f.write(f"{category_id} " + " ".join(map(str, yolo_bbox)) + '\n')
        except Exception as e:
            print(f"Failed to write to file {txt_file}: {e}")


def main() -> None:
    try:
        annotations = fetch_all_annotations()
        save_annotations(annotations, DetectionPaths.labels_input)
    except Exception as e:
        print(f"Failed to fetch annotations or save them: {e}")

if __name__ == "__main__":
    main()
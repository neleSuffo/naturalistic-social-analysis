import json
import logging
import cv2
from pathlib import Path
from utils import fetch_all_annotations
from multiprocessing import Pool
from constants import DetectionPaths, ClassificationPaths
from config import YoloConfig, CategoryMappings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_image_dimensions(image_paths: list) -> dict:
    """Preload image dimensions to avoid multiple cv2.imread() calls.
    
    Parameters
    ----------
    image_paths : list
        List of image file paths.
    
    Returns
    -------
    dict        
        Dictionary mapping image paths to their dimensions (height, width).
    """
    dimensions = {}
    for image_path in image_paths:
        if not image_path.exists():
            logging.warning(f"Image file not found: {image_path}")
            continue
            
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                logging.warning(f"Failed to load image: {image_path}")
                continue
            
            dimensions[image_path] = img.shape[:2]
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")
            continue
            
    if not dimensions:
        logging.error("No valid images found!")
        
    return dimensions

def convert_to_yolo_format(img_width: int, img_height: int, bbox: list) -> tuple:
    """Converts bounding box to YOLO format.
    
    Parameters
    ----------
    img_width : int
        Width of the image.
    img_height : int
        Height of the image
    bbox : list
        Bounding box coordinates [xtl, ytl, xbr, ybr].
        
    Returns
    -------
    tuple
        YOLO formatted bounding box (x_center, y_center, width, height).
    """
    # Convert bounding box to YOLO format
    xtl, ytl, xbr, ybr = bbox

    x_center = (xtl + xbr) / 2.0 / img_width
    y_center = (ytl + ybr) / 2.0 / img_height
    width = (xbr - xtl) / img_width
    height = (ybr - ytl) / img_height

    return (x_center, y_center, width, height)

def map_category_id(target: str, category_id: int, person_age: None, gaze_directed_at_child: None, object_interaction: None) -> int:
    """Maps category ID based on target type and additional attributes.
    
    Parameters
    ----------
    target : str
        The target detection type (e.g., "person", "face", "object").
    category_id : int
        The original category ID.
    person_age : None
        The age of the person (if applicable).
    gaze_directed_at_child : None
        The gaze direction (if applicable).
    object_interaction : None
        The object interaction type (if applicable).
    
    Returns
    -------
    int
        The mapped category ID.
    """
    mappings = {
        "gaze": CategoryMappings.gaze_cls.get(gaze_directed_at_child, 99),
        "person": CategoryMappings.person_cls.get(person_age, 99),
        "face": CategoryMappings.face_cls.get(person_age, 99),
        "object": CategoryMappings.object_det.get((category_id, object_interaction), 99),
        "person_face": CategoryMappings.person_face_det.get(category_id, 99),
        "person_face_object": CategoryMappings.person_face_object_det.get(category_id, 99),
        
    }
    return mappings.get(target, 99)

def write_annotations(txt_file: Path, lines: list) -> None:
    """Write annotation lines to a text file.
    
    Parameters
    ----------
    txt_file : Path
        Path to the output text file
    lines : list
        List of annotation lines to write
    """
    with open(txt_file, "w") as f:
        f.writelines(lines)
            
def save_annotations(annotations, target):
    """Saves annotations in YOLO format using optimized batch writing and multiprocessing."""
    logging.info("Saving annotations in YOLO format.")

    output_dirs = {
        "person_face": DetectionPaths.person_face_labels_input_dir,
        "person_face_object": DetectionPaths.person_face_object_labels_input_dir,
        "object": DetectionPaths.object_labels_input_dir,
        "person": ClassificationPaths.person_labels_input_dir,
        "face": ClassificationPaths.face_labels_input_dir,
        "gaze": ClassificationPaths.gaze_labels_input_dir,
    }

    if target not in output_dirs:
        raise ValueError(f"Invalid target: {target}. Must be one of: {', '.join(output_dirs.keys())}")

    output_dir = output_dirs[target]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preload image dimensions
    image_paths = {DetectionPaths.images_input_dir / ann[3][:-11] / ann[3] for ann in annotations}
    image_dims = get_image_dimensions(image_paths)

    if len(image_dims) == 0:
        raise RuntimeError("No valid images found. Please check the image paths and files.")
        
    file_contents = {}
    skipped_count = 0
    processed_count = 0

    for category_id, bbox_json, object_interaction, image_file_name, gaze_directed_at_child, person_age in annotations:
        image_file_path = DetectionPaths.images_input_dir / image_file_name[:-11] / image_file_name

        if image_file_path not in image_dims:
            skipped_count += 1
            continue

        bbox = json.loads(bbox_json)  # Parse JSON only once
        category_id = map_category_id(target, category_id, person_age, gaze_directed_at_child, object_interaction)
        img_height, img_width = image_dims[image_file_path]

        try:
            yolo_bbox = convert_to_yolo_format(img_width, img_height, bbox)
            line = f"{category_id} " + " ".join(map(str, yolo_bbox)) + "\n"

            if image_file_name not in file_contents:
                file_contents[image_file_name] = []
            file_contents[image_file_name].append(line)
            processed_count += 1
        except Exception as e:
            logging.error(f"Error converting bbox for {image_file_path}: {e}")
            skipped_count += 1
            continue

    write_tasks = []
    for img, lines in file_contents.items():
        output_path = output_dir / f"{Path(img).stem}.txt"
        write_tasks.append((output_path, lines))
        
    with Pool(processes=4) as pool:
        pool.starmap(write_annotations, write_tasks)

    logging.info(f"Processed {processed_count} annotations, skipped {skipped_count}.")

def main(target):
    """Main function to fetch and save YOLO annotations."""
    logging.info(f"Starting conversion process for YOLO {target} detection.")

    try:
        category_ids = {
            "person_face": YoloConfig.person_face_target_class_ids,
            "person_face_object": YoloConfig.person_face_object_target_class_ids,
            "object": YoloConfig.object_target_class_ids,
            "person": YoloConfig.person_target_class_ids,
            "face": YoloConfig.face_target_class_ids,
            "gaze": YoloConfig.face_target_class_ids,
        }.get(target)

        if category_ids is None:
            logging.error(f"Invalid target: {target}. Expected one of: {', '.join(category_ids.keys())}.")
            return

        annotations = fetch_all_annotations(category_ids=category_ids, persons=True, objects=(target == "object"), yolo_target=target)

        logging.info(f"Fetched {len(annotations)} {target} annotations.")
        save_annotations(annotations, target)
        logging.info(f"Successfully saved all {target} annotations.")

    except Exception as e:
        logging.error(f"Failed to process annotations: {e}")

if __name__ == "__main__":
    main()

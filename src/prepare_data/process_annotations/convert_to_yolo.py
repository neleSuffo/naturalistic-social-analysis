import json
import logging
import cv2
from pathlib import Path
from utils import fetch_all_annotations
from constants import DetectionPaths, ClassificationPaths
from config import YoloConfig, CategoryMappings

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

    # Calculate center coordinates
    x_center = (xtl + xbr) / 2.0
    y_center = (ytl + ybr) / 2.0
    
    # Calculate width and height
    width = xbr - xtl
    height = ybr - ytl
    
    # Normalize by image dimensions
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    # Return in YOLO format
    return (x_center, y_center, width, height)

def map_category_id(target: str, category_id: int, person_age: str = None, gaze_directed_at_child: str = None, object_interaction: str = None) -> int:
    """
    Maps category ID based on target type and additional attributes using CategoryMappings class.
    
    Parameters
    ----------
    target : str
        Target type for the mapping
    category_id : int
        Original category ID
    person_age : str, optional
        Age group for person categories
    gaze_directed_at_child : str, optional
        Gaze direction for gaze detection
    object_interaction : str, optional
        Interaction type for object detection
        
    Returns
    -------
    int
        Mapped category ID
    """
    # Get the mapping based on target
    
    # mapping for classification models
    if target == "gaze":
        return CategoryMappings.gaze.get(gaze_directed_at_child, 99)
    elif target in ["person", "face"]:
        mapping = CategoryMappings.person_face_cls
    
    # mapping for detection models
    elif target == "person_face":
        mapping = CategoryMappings.person_face_det
    elif target == "person_face_object":
        return CategoryMappings.person_face_object.get(category_id, 99)
    elif target == "object":
        return CategoryMappings.objects.get((category_id, object_interaction), 99)
    elif target == "all":
        mapping = CategoryMappings.all_instances
    elif target == "adult_person_face":
        mapping = CategoryMappings.adult_person_face
    elif target == "child_person_face":
        mapping = CategoryMappings.child_person_face
    else:
        logging.error(f"Invalid target: {target}")
        return 99
    
    # For person categories in adult/child detection, consider age
    if target in ["adult_person_face", "child_person_face", "all", "person", "face"] and person_age and category_id in [1, 2, 10]:
        return mapping.get((category_id, person_age.lower()), 99)
    
    # For all other cases, use direct mapping
    return mapping.get(category_id, 99)

def save_annotations(
    annotations: list,
    target: str
) -> None:
    """
    This function saves the annotations in YOLO format to text files.

    Parameters
    ----------
    annotations : list
        the list of annotations
    target : str
        the target detection type
    """
    logging.info("Saving annotations in YOLO format.")
    output_dirs = {
        "person_face": DetectionPaths.person_face_labels_input_dir,
        "person_face_object": DetectionPaths.person_face_object_labels_input_dir,
        "object": DetectionPaths.object_labels_input_dir,
        "person": ClassificationPaths.person_labels_input_dir,
        "face": ClassificationPaths.face_labels_input_dir,
        "gaze": ClassificationPaths.gaze_labels_input_dir,
    }
    # Remove default fallback and add error handling
    if target not in output_dirs:
        raise ValueError(f"Invalid target: {target}. Must be one of: {', '.join(output_dirs.keys())}")
    
    output_dir = output_dirs[target]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_contents = {}
    skipped_count = 0
    processed_count = 0
    #(category_id, bbox, object_interaction, image_file_name, gaze_directed_at_child, person_age)

    for annotation in annotations:    
        category_id, bbox, object_interaction, image_file_name, gaze_directed_at_child, person_age = annotation
        video_name = image_file_name[:-11]
        image_file_path = DetectionPaths.images_input_dir / video_name / image_file_name

        # Skip if image doesn't exist
        if not image_file_path.is_file():
            logging.warning(f"Image file {image_file_path} does not exist. Skipping annotation.")
            skipped_count += 1
            continue
        
        bbox = json.loads(bbox)
        
        category_id = map_category_id(target, category_id, person_age, gaze_directed_at_child, object_interaction)
        
        # YOLO format: category_id x_center y_center width height
        try:
            yolo_bbox = convert_to_yolo_format(image_file_path, bbox)               
            line = f"{category_id} " + " ".join(map(str, yolo_bbox)) + '\n'
            # Append the line to the list of lines for the image file
            if image_file_name not in file_contents:
                file_contents[image_file_name] = []
            file_contents[image_file_name].append(line)
            processed_count += 1
        except Exception as e:
            logging.error(f"Error converting bbox to YOLO format for {image_file_path}: {e}")
            skipped_count += 1
            continue

    # Write the lines to text files
    for image_file_name, lines in file_contents.items():
        file_name_without_extension = Path(image_file_name).stem
        txt_file = output_dir / (file_name_without_extension + '.txt')
        try:
            with open(txt_file, 'w') as f:
                f.writelines(lines)
        except IOError as e:
            logging.error(f"Failed to write to file {txt_file}: {e}")
    logging.info(f"Processed {processed_count} annotations, skipped {skipped_count} annotations")
    
def main(target: str) -> None:
    logging.info(f"Starting the conversion process for Yolo {target} detection.")
    try:
        category_ids = {
            "all": YoloConfig.all_target_class_ids,
            "person_face": YoloConfig.child_target_class_ids,
            "person_face_object": YoloConfig.all_target_class_ids,
            "adult_person_face": YoloConfig.adult_target_class_ids,
            "child_person_face": YoloConfig.child_target_class_ids,
            "object": YoloConfig.object_target_class_ids,
            "person": YoloConfig.person_target_class_ids,
            "face": YoloConfig.face_target_class_ids,
            "gaze": YoloConfig.face_target_class_ids,
        }.get(target)

        if category_ids is None:
            logging.error(f"Invalid target: {target}. Expected 'all', 'adult_person_face', 'child_person_face', 'object', 'person', 'face', 'gaze'.")
            return

        if target in ["all", "person_face_object"]:
            annotations = fetch_all_annotations(category_ids=category_ids, persons = True, objects=True)
        elif target in ["adult_person_face", "child_person_face", "gaze", "person", "face", "person_face"]:
            annotations = fetch_all_annotations(category_ids=category_ids, persons = True, objects=False, yolo_target=target)
        elif target == "object":
            annotations = fetch_all_annotations(category_ids=category_ids, persons = False, objects=False, yolo_target=target)
        logging.info(f"Fetched {len(annotations)} {target} annotations.")
        save_annotations(annotations, target)
        logging.info(f"Successfully saved all {target} annotations.")
    except Exception as e:
        print(f"Failed to fetch annotations or save them: {e}")

if __name__ == "__main__":
    main()

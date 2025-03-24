import json
import logging
import cv2
from pathlib import Path
from utils import fetch_all_annotations
from constants import YoloPaths, DetectionPaths
from config import YoloConfig

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


def find_alternative_image(
    image_file_name: str,
    video_name: str
) -> Path:
    """ 
    This function searches for an alternative image file with the same quantex_at_home_id.
    
    Parameters
    ----------
    image_file_name : str
        The original image file name.
    video_name: str
        the name of the video the image belongs to

    Returns
    -------
    Path
        The path to the alternative image file, or None if not found.
    """
    # Split the string by underscores
    parts = image_file_name.split('_')
    # Join the relevant parts (up to the 5th underscore)
    quantex_id = "_".join(parts[:8])
    
    # Search for alternative images in the directory with the same quantex_at_home_id
    for image_path in DetectionPaths.images_input_dir/video_name.glob(f"{quantex_id}*"):
        if image_path.name != image_file_name:
            return image_path
    
    return None


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
    output_dir =  YoloPaths.all_labels_input_dir if target == "all" else YoloPaths.person_face_labels_input_dir if target == "person_face" else YoloPaths.person_face_object_labels_input_dir if target == "person_face_object" else YoloPaths.gaze_labels_input_dir      
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_contents = {}
    skipped_count = 0
    processed_count = 0
    #(image_id, video_id, category_id, bbox, image_file_name, video_file_name, gaze_directed_at_child, person_age)

    for annotation in annotations:           
        _, _, category_id, bbox, image_file_name, _, gaze_directed_at_child, person_age = annotation
        video_name = image_file_name[:-11]
        image_file_path = DetectionPaths.images_input_dir / video_name / image_file_name

        # Skip if image doesn't exist
        if not image_file_path.is_file():
            logging.warning(f"Image file {image_file_path} does not exist. Skipping annotation.")
            skipped_count += 1
            continue
        
        bbox = json.loads(bbox)
        
        # define mapping for the category_id
        gaze_mapping = {'No': 0, 'Yes': 1}
        child_person_face_mapping =   {(1, 'Inf'): 0, (1, 'Child'): 0, (1, 'Teen'): 99, (10, 'Adult'): 99, 
                        (2, 'Inf'): 0, (2, 'Child'): 0, (2, 'Teen'): 99, (2, 'Adult'): 99, 
                        (10, 'infant'): 1, (10, 'child'): 1, (10, 'teen'): 99, (10, 'adult'): 99,
                        11: 2, 3: 99, 4: 99, 5: 99, 6: 99, 7: 99, 8: 99, 12: 99}
        adult_person_face_mapping =   {(1, 'Inf'): 99, (1, 'Child'): 99, (1, 'Teen'): 0, (10, 'Adult'): 0, 
                (2, 'Inf'): 99, (2, 'Child'): 99, (2, 'Teen'): 0, (2, 'Adult'): 0, 
                (10, 'infant'): 99, (10, 'child'): 99, (10, 'teen'): 1, (10, 'adult'): 1,
                11: 2, 3: 99, 4: 99, 5: 99, 6: 99, 7: 99, 8: 99, 12: 99}
        person_face_mapping = {1: 0, 2: 0, 10: 1, 11:2}
        person_face_object_mapping = {1: 0, 2: 0, 10: 1, 11:2, 3:3, 4:9, 5:4, 6:5, 7:6, 8:7, 12:8} #map class 4 (animal) to 9, to be able to exclude it later
        all_mapping =   {(1, 'Inf'): 0, (1, 'Child'): 0, (1, 'Teen'): 1, (10, 'Adult'): 1, 
                        (2, 'Inf'): 0, (2, 'Child'): 0, (2, 'Teen'): 1, (2, 'Adult'): 1, 
                        (10, 'infant'): 2, (10, 'child'): 2, (10, 'teen'): 3, (10, 'adult'): 3,
                        11: 4, 3: 5, 4: 11, 5: 6, 6: 7, 7: 8, 8: 9, 12: 10}
        
        if target == "gaze":
            # category id is replaced with gaze_directed_at_child (No: 0, Yes: 1)
            category_id = gaze_mapping.get(gaze_directed_at_child, gaze_directed_at_child)
        elif target == "all":
            # map person based on age group
            category_id = all_mapping.get((category_id, person_age), all_mapping.get(category_id, category_id))            
        elif target == "person_face":
            # Map the category_id to the YOLO format (treat person, reflection, face all as category "person")
            category_id = person_face_mapping.get(category_id, category_id)
        elif target == "person_face_object":
            # Map the category_id to the YOLO format (treat person, reflection, face all as category "person")
            category_id = person_face_object_mapping.get(category_id, category_id)
        elif target == "adult_person_face":
            # Map the category_id to the YOLO format 
            category_id = adult_person_face_mapping.get((category_id, person_age), category_id)
        elif target == "child_person_face":
            # Map the category_id to the YOLO format 
            category_id = child_person_face_mapping.get((category_id, person_age), category_id)
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
            "adult_person_face": YoloConfig.person_target_class_ids,
            "child_person_face": YoloConfig.person_target_class_ids,
            "person_face": YoloConfig.person_target_class_ids,
            "person_face_object": YoloConfig.person_object_target_class_ids,
            "gaze": YoloConfig.face_target_class_ids,
            "all": YoloConfig.person_object_target_class_ids,
        }.get(target)

        if category_ids is None:
            logging.error(f"Invalid target: {target}. Expected 'all', 'person_face', 'person_face_object' or 'gaze'.")
            return

        if target == "person_face_object" or target == "all":
            annotations = fetch_all_annotations(category_ids=category_ids, objects=True)
        else:
            annotations = fetch_all_annotations(category_ids=category_ids)
        logging.info(f"Fetched {len(annotations)} {target} annotations.")
        save_annotations(annotations, target)
        logging.info(f"Successfully saved all {target} annotations.")
    except Exception as e:
        print(f"Failed to fetch annotations or save them: {e}")

if __name__ == "__main__":
    main()

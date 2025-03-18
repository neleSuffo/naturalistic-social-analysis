import cv2
import logging
import argparse
import os
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from constants import Proximity, YoloPaths

# Configure logging
logging.basicConfig(level=logging.INFO)

def extract_bounding_boxes(results):
    """Extract bounding boxes for children, adults, and their faces."""
    child, adult, child_face, adult_face = [], [], [], []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = (x1, y1, x2 - x1, y2 - y1)  # (x, y, width, height)
            if cls == 0:
                child.append(bbox)
            elif cls == 1:
                adult.append(bbox)
            elif cls == 2:
                child_face.append(bbox)
            elif cls == 3:
                adult_face.append(bbox)
    return child, adult, child_face, adult_face

def find_matching_person(face_bbox, person_bboxes):
    """Find the person bounding box that contains the face."""
    x_face, y_face, w_face, h_face = face_bbox
    face_center = (x_face + w_face // 2, y_face + h_face // 2)
    
    for person_bbox in person_bboxes:
        x_person, y_person, w_person, h_person = person_bbox
        if (x_person <= face_center[0] <= x_person + w_person and 
            y_person <= face_center[1] <= y_person + h_person):
            return person_bbox
    return None

def normalize_proximity(face_area, ref_far, ref_close):
    """Normalize proximity so that ref_far maps to 0 and ref_close maps to 1."""
    if ref_close is None or ref_far is None or ref_close == ref_far:
        logging.warning("Invalid reference values for proximity calculation.")
        return None

    normalized_value = (face_area - ref_far) / (ref_close - ref_far)
    return max(0, min(1, normalized_value))  # Ensure it's between 0 and 1

def calculate_proximity(face_bbox, min_ref_area, max_ref_area, ref_aspect_ratio, aspect_ratio_threshold=0.5):
    """
    Compute proximity based on detected face area and aspect ratio relative to reference values.
    
    Parameters:
    ----------
    face_bbox: tuple
        Bounding box coordinates for the detected face (x, y, width, height)
    min_ref_area: int
        Minimum reference area for face detection
    max_ref_area: int
        Maximum reference area for face detection
    ref_aspect_ratio: float 
        Reference aspect ratio for face detection
    aspect_ratio_threshold: float
        Maximum deviation from the reference aspect ratio
        
    Returns:
    --------
    float
        Proximity value for the detected face
    
    """
    if face_bbox is None:
        return None

    # Extract face bounding box coordinates
    x1, y1, x2, y2 = face_bbox
    face_width = x2 - x1
    face_height = y2 - y1
    face_area = face_width * face_height
    aspect_ratio = float(face_width) / float(face_height)

    print(f"detected face_area: {face_area}")

    # Check if the face is a "partial face" based on aspect ratio
    if ref_aspect_ratio is not None:  # Only check if we have a valid reference ratio
        if abs(aspect_ratio - ref_aspect_ratio) > aspect_ratio_threshold:
            logging.info(f"Partial face detected! Aspect ratio {aspect_ratio:.2f} deviates significantly from reference {ref_aspect_ratio:.2f}")
            return 1.0  # Partial face: very close

    # Normalize face area to a value between 0 and 1
    if face_area <= max_ref_area:
        logging.info("Face area smaller than max_ref_area")
        return 0.0  # Smallest possible proximity (far away)
    if face_area >= min_ref_area:
        logging.info("Face area larger than min_ref_area")
        return 1.0  # Largest possible proximity (very close)

    # Scale the face area between min_ref_area and max_ref_area
    #proximity = (face_area - max_ref_area) / (min_ref_area - max_ref_area)  
    proximity = (np.log(face_area) - np.log(max_ref_area)) / (np.log(min_ref_area) - np.log(max_ref_area)) 
    return proximity

# File to store reference values
def save_reference_metrics(metrics: dict, reference_file: Path = Proximity.reference_file):
    """Save reference metrics to a JSON file."""
    with open(reference_file, "w") as f:
        json.dump(metrics, f)

def load_reference_metrics(reference_file: Path = Proximity.reference_file):
    """Load reference metrics from a JSON file if available."""
    if os.path.exists(reference_file):
        with open(reference_file, "r") as f:
            data = json.load(f)
        # Ensure all keys are present for backward compatibility
        if not all(k in data for k in ("child_ref_close", "child_ref_far", "adult_ref_close", "adult_ref_far", "child_ref_aspect_ratio", "adult_ref_aspect_ratio")):
            logging.warning("Reference file is missing some metrics, recomputing...")
            return None
        return data
    return None

def get_reference_proximity_metrics(model, child_close_image_path, child_far_image_path, adult_close_image_path, adult_far_image_path):
    """Retrieve or compute reference proximity metrics and aspect ratios separately for child and adult faces."""

    # Try loading stored references
    stored_metrics = load_reference_metrics()
    if stored_metrics:
        logging.info("Loaded reference metrics from file.")
        return (stored_metrics["child_ref_close"], stored_metrics["child_ref_far"],
                stored_metrics["adult_ref_close"], stored_metrics["adult_ref_far"],
                stored_metrics["child_ref_aspect_ratio"], stored_metrics["adult_ref_aspect_ratio"])

    logging.info("Computing reference metrics...")

    child_close_image = cv2.imread(child_close_image_path)
    child_far_image = cv2.imread(child_far_image_path)
    adult_close_image = cv2.imread(adult_close_image_path)
    adult_far_image = cv2.imread(adult_far_image_path)

    results_child_close = model(child_close_image)
    results_child_far = model(child_far_image)
    results_adult_close = model(adult_close_image)
    results_adult_far = model(adult_far_image)

    _, _, child_face_close, _ = extract_bounding_boxes(results_child_close)
    _, _, child_face_far, _ = extract_bounding_boxes(results_child_far)
    _, _, _, adult_face_close = extract_bounding_boxes(results_adult_close)
    _, _, _, adult_face_far = extract_bounding_boxes(results_adult_far)

    def compute_average_area(faces, label):
        if not faces:
            logging.warning(f"No {label} faces detected in reference images.")
            return None
        return sum([face[2] * face[3] for face in faces]) / len(faces)

    def compute_average_aspect_ratio(faces, label):
        if not faces:
            logging.warning(f"No {label} faces detected in reference images.")
            return None
        aspect_ratios = [float(face[2]) / float(face[3]) for face in faces]  # width / height
        return sum(aspect_ratios) / len(aspect_ratios)

    child_ref_close = compute_average_area(child_face_close, "child close")
    child_ref_far = compute_average_area(child_face_far, "child far")
    adult_ref_close = compute_average_area(adult_face_close, "adult close")
    adult_ref_far = compute_average_area(adult_face_far, "adult far")

    child_ref_aspect_ratio = compute_average_aspect_ratio(child_face_close, "child close")
    adult_ref_aspect_ratio = compute_average_aspect_ratio(adult_face_close, "adult close")

    logging.info(f"Child Ref Close: {child_ref_close}, Child Ref Far: {child_ref_far}")
    logging.info(f"Adult Ref Close: {adult_ref_close}, Adult Ref Far: {adult_ref_far}")
    logging.info(f"Child Ref Aspect Ratio: {child_ref_aspect_ratio}, Adult Ref Aspect Ratio: {adult_ref_aspect_ratio}")

    # Save computed references for future runs
    metrics = {
        "child_ref_close": child_ref_close,
        "child_ref_far": child_ref_far,
        "adult_ref_close": adult_ref_close,
        "adult_ref_far": adult_ref_far,
        "child_ref_aspect_ratio": child_ref_aspect_ratio,
        "adult_ref_aspect_ratio": adult_ref_aspect_ratio
    }
    save_reference_metrics(metrics)

    return (child_ref_close, child_ref_far, adult_ref_close, adult_ref_far,
            child_ref_aspect_ratio, adult_ref_aspect_ratio)


def describe_proximity(proximity):
    """Returns a qualitative description of proximity."""
    if proximity is None:
        return "No valid detection"
    elif proximity < 0.2:
        return "Further away"
    elif proximity < 0.4:
        return "Quite far"
    elif proximity < 0.6:
        return "Moderate distance"
    elif proximity < 0.8:
        return "Quite close"
    else:
        return "Very close"

def compute_proximity(image_path, model, ref_metrics):
    """
    Compute and return the proximity value for each detected face in a given image.
    
    Parameters:
    ----------
    image_path: str
        Path to the input image.
    model: YOLO
        YOLO model for face detection.
    ref_metrics: tuple
        Reference metrics for proximity calculation.
    """
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image from {image_path}")
        return {}

    results = model(image)
    child_bboxes, adult_bboxes, child_faces, adult_faces = extract_bounding_boxes(results)

    proximities = {}  # Store proximity values for each detected face
    logging.info(f"Proximity values for {image_path}")

    # Process child faces
    for i, face_bbox in enumerate(child_faces):
        proximity = calculate_proximity(face_bbox, ref_metrics[0], ref_metrics[1], ref_metrics[4]) # Pass child aspect ratio
        if proximity is not None:
            description = describe_proximity(proximity)
            face_key = f"child_face_{i+1}"
            proximities[face_key] = proximity
            logging.info(f"{face_key}: Proximity = {proximity:.2f} ({description})")

    # Process adult faces
    for i, face_bbox in enumerate(adult_faces):
        proximity = calculate_proximity(face_bbox, ref_metrics[2], ref_metrics[3], ref_metrics[5]) # Pass adult aspect ratio
        if proximity is not None:
            description = describe_proximity(proximity)
            face_key = f"adult_face_{i+1}"
            proximities[face_key] = proximity
            logging.info(f"{face_key}: Proximity = {proximity:.2f} ({description})")

    if not proximities:
        logging.info(f"No valid proximity data for image {image_path}")

def return_proximity(bounding_box: list, face_type: str, ref_metrics: tuple):
    """
    Compute and return the proximity value for each detected face in a given image.
    
    Parameters:
    ----------
    bounding_box: list
        Bounding box coordinates for the detected face
    face_type: str
        Type of the detected face (child or adult)
    ref_metrics: tuple
        Reference metrics for proximity calculation.
        
    Returns:
    --------
    dict
        Proximity value for the detected face
    
    """
    if face_type == "infant/child face":
        proximity = calculate_proximity(bounding_box, ref_metrics[0], ref_metrics[1], ref_metrics[4])
        print("proximity", proximity)
        return proximity
    elif face_type == "adult face":
        proximity = calculate_proximity(bounding_box, ref_metrics[2], ref_metrics[3], ref_metrics[5])
        print("proximity", proximity)
        return proximity
        
def get_proximity(bounding_box: list, face_type: str):
    """ 
    This function is used to compute the proximity of detected faces in an image.
    
    Parameters:
    ----------
    bounding_box: list
        Bounding box coordinates for the detected face
    face_type: str
        Type of the detected face (child or adult)
    
    Returns:
    --------
    proximity: float
        Proximity value for the detected face
    
    """
    ref_metrics = load_reference_metrics()
    if ref_metrics is None:
        logging.warning("Reference metrics not found. Please run the reference computation script.")
        return
    ref_metrics_list = list(ref_metrics.values())
    proximity = return_proximity(bounding_box, face_type, ref_metrics_list)
    return proximity

def main():
    parser = argparse.ArgumentParser(description='Compute proximity for detected faces in an image.')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to the input image')
    args = parser.parse_args()

    model_path = YoloPaths.all_trained_weights_path
    child_close_image_path = Proximity.child_close_image_path
    child_far_image_path = Proximity.child_far_image_path
    adult_close_image_path = Proximity.adult_close_image_path
    adult_far_image_path = Proximity.adult_far_image_path

    model = YOLO(model_path)
    (child_ref_close, child_ref_far, adult_ref_close, adult_ref_far,
     child_ref_aspect_ratio, adult_ref_aspect_ratio) = get_reference_proximity_metrics(model, child_close_image_path, child_far_image_path, adult_close_image_path, adult_far_image_path)
    ref_metrics = (child_ref_close, child_ref_far, adult_ref_close, adult_ref_far, child_ref_aspect_ratio, adult_ref_aspect_ratio)
    compute_proximity(args.image_path, model, ref_metrics)


if __name__ == "__main__":
    main()
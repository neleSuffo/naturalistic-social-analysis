import cv2
import logging
import os
import json
from ultralytics import YOLO

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

def normalize_proximity(raw_proximity, min_proximity, max_proximity):
    """Normalize proximity between 0 and 1, where 1 indicates closeness."""
    return max(0, min(1, 1 - (raw_proximity - min_proximity) / (max_proximity - min_proximity)))

def calculate_proximity(face_bbox, person_bbox, image_shape, ref_close, ref_far):
    """Calculate normalized proximity based on bounding box size and reference values."""
    if ref_close is None or ref_far is None:
        return None  # Cannot compute without references

    face_area = face_bbox[2] * face_bbox[3]  # width * height
    proximity_metric = (face_area - ref_far) / (ref_close - ref_far)
    return normalize_proximity(proximity_metric, 0, 1)

def process_detections(faces, persons, image_shape, ref_close, ref_far):
    """Process detections and compute proximity information."""
    proximities = []
    for face_bbox in faces:
        matching_person = find_matching_person(face_bbox, persons)
        proximity = calculate_proximity(face_bbox, matching_person, image_shape, ref_close, ref_far)
        if proximity is not None:
            proximities.append(proximity)
    
    if proximities:
        return sum(proximities) / len(proximities)  # Return the average proximity
    return None

# File to store reference values
REFERENCE_FILE = "/home/nele_pauline_suffo/outputs/reference_proximity.json"

def save_reference_metrics(metrics):
    """Save reference metrics to a JSON file."""
    with open(REFERENCE_FILE, "w") as f:
        json.dump(metrics, f)

def load_reference_metrics():
    """Load reference metrics from a JSON file if available."""
    if os.path.exists(REFERENCE_FILE):
        with open(REFERENCE_FILE, "r") as f:
            return json.load(f)
    return None

def get_reference_proximity_metrics(model, child_close_image_path, child_far_image_path, adult_close_image_path, adult_far_image_path):
    """Retrieve or compute reference proximity metrics separately for child and adult faces."""
    
    # Try loading stored references
    stored_metrics = load_reference_metrics()
    if stored_metrics:
        logging.info("Loaded reference metrics from file.")
        return stored_metrics["child_ref_close"], stored_metrics["child_ref_far"], stored_metrics["adult_ref_close"], stored_metrics["adult_ref_far"]
    
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

    child_ref_close = compute_average_area(child_face_close, "child close")
    child_ref_far = compute_average_area(child_face_far, "child far")
    adult_ref_close = compute_average_area(adult_face_close, "adult close")
    adult_ref_far = compute_average_area(adult_face_far, "adult far")

    logging.info(f"Child Ref Close: {child_ref_close}, Child Ref Far: {child_ref_far}")
    logging.info(f"Adult Ref Close: {adult_ref_close}, Adult Ref Far: {adult_ref_far}")

    # Save computed references for future runs
    metrics = {
        "child_ref_close": child_ref_close,
        "child_ref_far": child_ref_far,
        "adult_ref_close": adult_ref_close,
        "adult_ref_far": adult_ref_far
    }
    save_reference_metrics(metrics)

    return child_ref_close, child_ref_far, adult_ref_close, adult_ref_far

def describe_proximity(proximity):
    """Returns a qualitative description of proximity."""
    if proximity is None:
        return "No valid detection"
    elif proximity < 0.2:
        return "Very far away"
    elif proximity < 0.4:
        return "Quite far"
    elif proximity < 0.6:
        return "Moderate distance"
    elif proximity < 0.8:
        return "Quite close"
    else:
        return "Very close"

def compute_proximity(image_path, model, ref_metrics):
    """Compute and return the proximity value for a given image."""
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image from {image_path}")
        return None

    results = model(image)
    child, adult, child_face, adult_face = extract_bounding_boxes(results)

    child_proximity = process_detections(child_face, child, image.shape, ref_metrics[0], ref_metrics[1])
    adult_proximity = process_detections(adult_face, adult, image.shape, ref_metrics[2], ref_metrics[3])

    # Combine proximity values
    proximities = [p for p in [child_proximity, adult_proximity] if p is not None]
    average_proximity = sum(proximities) / len(proximities) if proximities else None

    # Get qualitative description
    proximity_description = describe_proximity(average_proximity)

    if average_proximity is not None:
        logging.info(f"Proximity for image {image_path}: {average_proximity:.2f} ({proximity_description})")
    else:
        logging.info(f"No valid proximity data for image {image_path}")

    return average_proximity

def main():
    model_path = "/home/nele_pauline_suffo/models/yolo11_all_detection.pt"
    child_close_image_path = "/home/nele_pauline_suffo/ProcessedData/quantex_videos_processed/quantex_at_home_id264041_2023_05_22_07/quantex_at_home_id264041_2023_05_22_07_041970.jpg"
    child_far_image_path = "/home/nele_pauline_suffo/ProcessedData/quantex_videos_processed/quantex_at_home_id255944_2022_03_25_01/quantex_at_home_id255944_2022_03_25_01_052500.jpg"
    adult_close_image_path = "/home/nele_pauline_suffo/ProcessedData/quantex_videos_processed/quantex_at_home_id264368_2024_10_18_01/quantex_at_home_id264368_2024_10_18_01_000060.jpg"
    adult_far_image_path = "/home/nele_pauline_suffo/ProcessedData/quantex_videos_processed/quantex_at_home_id264683_2024_09_28_01/quantex_at_home_id264683_2024_09_28_01_001620.jpg"

    image_path = "/home/nele_pauline_suffo/ProcessedData/quantex_videos_processed/quantex_at_home_id267094_2023_04_06_02/quantex_at_home_id267094_2023_04_06_02_023970.jpg"

    model = YOLO(model_path)
    ref_metrics = get_reference_proximity_metrics(model, child_close_image_path, child_far_image_path, adult_close_image_path, adult_far_image_path)
    proximity = compute_proximity(image_path, model, ref_metrics)

if __name__ == "__main__":
    main()
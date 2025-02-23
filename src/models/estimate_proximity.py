import cv2
import logging
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)

def extract_bounding_boxes(results):
    """
    Extract all face and person bounding boxes from the inference results.
    
    Parameters:
        results (list): Inference results from the YOLO model.
    
    Returns:
        faces (list): List of face bounding boxes [(x, y, width, height), ...].
        persons (list): List of person bounding boxes [(x, y, width, height), ...].
    """
    faces = []
    persons = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = (x1, y1, x2 - x1, y2 - y1)

            if cls == 0:
                persons.append(bbox)
            elif cls == 1:
                faces.append(bbox)

    return faces, persons

def find_matching_person(face_bbox, person_bboxes):
    """
    Find the person bounding box that contains the face.
    
    Parameters:
        face_bbox (tuple): (x, y, width, height) of the face
        person_bboxes (list): List of person bounding boxes
    
    Returns:
        matching_person (tuple): Matching person bbox or None if no match
    """
    x_face, y_face, w_face, h_face = face_bbox
    face_center = (x_face + w_face//2, y_face + h_face//2)

    for person_bbox in person_bboxes:
        x_person, y_person, w_person, h_person = person_bbox
        
        # Check if face center is inside person bbox
        if (x_person <= face_center[0] <= x_person + w_person and 
            y_person <= face_center[1] <= y_person + h_person):
            return person_bbox
            
    return None

def process_detections(faces, persons, image_shape):
    """
    Process all face detections and find their corresponding person boxes.
    
    Returns:
        list: List of (face_bbox, person_bbox, proximity, full_face_visible) tuples
    """
    results = []
    
    for face_bbox in faces:
        matching_person = find_matching_person(face_bbox, persons)
        proximity, full_face_visible = calculate_proximity(
            face_bbox, matching_person, image_shape
        )
        results.append((face_bbox, matching_person, proximity, full_face_visible))
        
    return results

# Function to normalize proximity
def normalize_proximity(raw_proximity, min_proximity, max_proximity):
    normalized = (raw_proximity - min_proximity) / (max_proximity - min_proximity)
    normalized = max(0, min(1, normalized))  # Clamp to [0, 1]
    return 1 - normalized  # Invert if 1 should indicate closeness

def calculate_proximity(face_bbox, person_bbox, image_shape, min_distance=0, max_distance=1.0):
    """
    Calculate normalized proximity based on face and person bounding boxes.

    Parameters:
        face_bbox (tuple): (x, y, width, height) of the face bounding box.
        person_bbox (tuple or None): (x, y, width, height) of the person bounding box or None.
        image_shape (tuple): Shape of the image (height, width, channels).
        min_distance (float): Expected minimum distance for normalization.
        max_distance (float): Expected maximum distance for normalization.

    Returns:
        normalized_proximity (float): Normalized proximity metric ranging from 0 (close) to 1 (far).
        full_face_visible (bool): Indicates if the full face is visible.
    """
    img_height, img_width, _ = image_shape

    # Calculate area of the face bounding box
    face_area = face_bbox[2] * face_bbox[3]

    if person_bbox:
        # Calculate area of the person bounding box
        person_area = person_bbox[2] * person_bbox[3]

        # Calculate proximity metric based on face area relative to person area
        if person_area > 0:
            proximity_metric = face_area / person_area  # Ratio of areas
        else:
            proximity_metric = 0  # Handle case where person area is zero

        # Check if the face bounding box is inside the person bounding box
        x_face, y_face, w_face, h_face = face_bbox
        x_person, y_person, w_person, h_person = person_bbox

        full_face_visible = (
            x_face >= x_person and
            y_face >= y_person and
            (x_face + w_face) <= (x_person + w_person) and
            (y_face + h_face) <= (y_person + h_person)
        )
    else:
        # If no person is detected, set proximity metric based on face size relative to image size
        image_area = img_width * img_height
        proximity_metric = face_area / image_area
        full_face_visible = False 

    # Normalize the proximity metric to range between 0 and 1
    normalized_proximity = (proximity_metric - min_distance) / (max_distance - min_distance)
    normalized_proximity = max(0, min(1, normalized_proximity))  # Clamp to [0, 1]

    # Invert the metric so that 0 is close and 1 is far
    normalized_proximity = 1 - normalized_proximity

    return normalized_proximity, full_face_visible

def draw_bounding_boxes(image, face_bbox, person_bbox):
    """
    Draw bounding boxes on the image for face and person detections.
    
    Parameters:
        image (numpy.ndarray): Image on which to draw.
        face_bbox (tuple): Bounding box for the face (x, y, width, height).
        person_bbox (tuple): Bounding box for the person (x, y, width, height).
    
    Returns:
        image (numpy.ndarray): Image with drawn bounding boxes.
    """
    if person_bbox:
        x1, y1 = person_bbox[0], person_bbox[1]
        x2, y2 = x1 + person_bbox[2], y1 + person_bbox[3]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box for person

    if face_bbox:
        x1, y1 = face_bbox[0], face_bbox[1]
        x2, y2 = x1 + face_bbox[2], y1 + face_bbox[3]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for face

    return image

def main():
    # Paths to model and images
    model_path = '/home/nele_pauline_suffo/models/yolo11_person_face_detection.pt'
    image_path = '/home/nele_pauline_suffo/ProcessedData/quantex_videos_processed/quantex_at_home_id262333_2024_12_01_05/quantex_at_home_id262333_2024_12_01_05_000150.jpg'
    output_path = '/home/nele_pauline_suffo/outputs/yolo_person_face_classification/proximity_heuristic.jpg'
    close_image_path = '/home/nele_pauline_suffo/ProcessedData/quantex_videos_processed/quantex_at_home_id255944_2022_03_10_01/quantex_at_home_id255944_2022_03_10_01_000000.jpg'
    far_image_path = '/home/nele_pauline_suffo/ProcessedData/quantex_videos_processed/quantex_at_home_id255944_2022_03_25_01/quantex_at_home_id255944_2022_03_25_01_045360.jpg'

    # Load the YOLO model
    model = YOLO(model_path)

    # Load and process reference images
    close_image = cv2.imread(close_image_path)
    far_image = cv2.imread(far_image_path)
    
    # Perform inference
    results_close = model(close_image)
    results_far = model(far_image)
    
    face_bbox_close, person_bbox_close = extract_bounding_boxes(results_close)
    face_bbox_far, person_bbox_far = extract_bounding_boxes(results_far)

    # Calculate raw proximity metrics for reference images
    min_proximity, _ = calculate_proximity(face_bbox_close, person_bbox_close, close_image.shape)
    max_proximity, _ = calculate_proximity(face_bbox_far, person_bbox_far, far_image.shape)
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image from {image_path}")

    # Perform inference
    results = model(image)

    # Extract bounding boxes
    faces, persons = extract_bounding_boxes(results)

    # Ensure at least a face is detected
    if faces:
        detections = process_detections(faces, persons, image.shape)

        for face_bbox, person_bbox, proximity, full_face_visible in detections:
            # Draw boxes for each face-person pair
            image = draw_bounding_boxes(image, face_bbox, person_bbox)
            
            # Add proximity text for each detection
            x, y = face_bbox[0], face_bbox[1] - 10
            cv2.putText(image, f"Proximity: {proximity:.2f}", 
                       (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Save the output image
        cv2.imwrite(output_path, image)
        logging.info(f"Output image saved to {output_path}")
        
if __name__ == "__main__":
    main()
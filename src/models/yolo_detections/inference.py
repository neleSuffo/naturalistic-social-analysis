import cv2
import logging
import argparse
import random
from ultralytics import YOLO
from constants import YoloPaths
from detection_pipeline.estimate_proximity import get_proximity
from config import YoloConfig

# Add this after the imports
CLASS_COLORS = {
    0: (215, 65, 117),    # Pink for class 0
    1: (105, 22, 51),  # Magenta for class 1
    2: (199, 200, 126),    # Green for class 2
    3: (210, 187, 109),  # Light Green for class 3
    4: (179, 182, 176),    # Gray for class 4
    5: (102, 120, 124),  # Strong Gray for class 5
    6: (141, 142, 61),  # Strong Green for class 6
    7: (45, 55, 58),      # Black for class 7
    8: (242, 192, 209),    # Light Pink for class 8
    9: (201, 210, 213),    # Ice Blue for class 9
    10: (217, 218, 169),   # Green Yellow for class 10
}
# Load the YOLOv11 model
model = YOLO(YoloPaths.all_trained_weights_path)

def custom_nms(results, iou_threshold):
    """ 
    This function performs a custom non-maximum suppression (NMS) on the detected boxes.
    It filters out boxes that have an IoU greater than the threshold with any other box.
    
    Parameters:
    ----------
    results : List
        A list of results from the YOLO model predictions.
    iou_threshold : float
        The threshold value for the Intersection over Union (IoU) metric.
        
    Returns:
    -------
    List
        A list of filtered boxes after applying the custom
        non-maximum suppression. (x1, y1, x2, y2, class_id, confidence_score)
    """
    # Collect all boxes into a single list
    all_boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers
            class_id = int(box.cls[0])  # Class ID
            confidence_score = float(box.conf[0])  # Confidence score
            all_boxes.append((x1, y1, x2, y2, class_id, confidence_score))
    
    # Sort boxes by confidence score in descending order
    all_boxes.sort(key=lambda x: x[5], reverse=True)
    
    # Initialize list to store filtered boxes
    filtered_boxes = []
    
    for box in all_boxes:
        x1, y1, x2, y2, class_id, confidence_score = box
        overlaps = []
        for existing_box in filtered_boxes:
            existing_x1, existing_y1, existing_x2, existing_y2, _, _ = existing_box
            overlap = calculate_iou((x1, y1, x2, y2), (existing_x1, existing_y1, existing_x2, existing_y2))
            overlaps.append(overlap)
        
        if not any(overlap > iou_threshold for overlap in overlaps):
            filtered_boxes.append(box)
    return filtered_boxes

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    intersection_area = max(0, min(x2, x4) - max(x1, x3)) * max(0, min(y2, y4) - max(y1, y3))
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area

def calculate_detection_scores(face_detection, person_detection, AP_values):
    """Calculate detection scores for face-person pair."""
    face_confidence = face_detection['confidence']
    person_confidence = person_detection['confidence']
    
    scores = {
        'adult_face': AP_values['adult_face'] * face_confidence,
        'child_face': AP_values['child_face'] * face_confidence,
        'adult_person': AP_values['adult'] * person_confidence,
        'child_person': AP_values['child'] * person_confidence
    }
    
    scores['adult_combined'] = scores['adult_face'] + scores['adult_person']
    scores['child_combined'] = scores['child_face'] + scores['child_person']
    
    return scores

def update_detection_classes(face_detection, person_detection, scores, model_names):
    """Update detection classes based on scores."""
    if scores['adult_combined'] > scores['child_combined']:
        face_detection.update({'class_id': 3, 'class_name': model_names[3]})
        person_detection.update({'class_id': 1, 'class_name': model_names[1]})
    else:
        face_detection.update({'class_id': 2, 'class_name': model_names[2]})
        person_detection.update({'class_id': 0, 'class_name': model_names[0]})

def is_face_inside_person(face_box, person_box):
    """Check if face bounding box is inside person bounding box."""
    face_x1, face_y1, face_x2, face_y2 = face_box
    person_x1, person_y1, person_x2, person_y2 = person_box
    return (person_x1 <= face_x1 and person_y1 <= face_y1 and
            person_x2 >= face_x2 and person_y2 >= face_y2)

def draw_detection(image, detection, proximity=None):
    """Draw bounding box and label for a detection."""
    x1, y1, x2, y2 = detection['box']
    class_id = detection['class_id']
    confidence = detection['confidence']
    class_name = detection['class_name']
    color = CLASS_COLORS[class_id]
    
    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
    
    # Create label
    label = (f"{class_name} {confidence:.2f}, Proximity: {proximity:.2f}" 
            if proximity is not None else f"{class_name} {confidence:.2f}")
    
    # Draw label background and text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(label, font, 0.5, 1)[0]
    text_x = x1
    text_y = y1 - 5 if y1 - 5 > 10 else y1 + 15
    
    cv2.rectangle(image, (text_x, text_y - text_size[1] - 2),
                 (text_x + text_size[0], text_y + 2), color, -1)
    cv2.putText(image, label, (text_x, text_y), font, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)

def run_inference(image_path):
    """Run object detection inference on an image."""
    # Load and verify image
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to read image: {image_path}")
        return
    
    # Define AP values
    AP_values = {
        'adult_face': 0.956,
        'adult': 0.947,
        'child_face': 0.856,
        'child': 0.863
    }
    
    # Run detection and NMS
    results = model.predict(image, iou=YoloConfig.best_iou)
    filtered_boxes = custom_nms(results, iou_threshold=YoloConfig.between_classes_iou)
    logging.info(f"Detected classes after NMS: {[model.names[box[4]] for box in filtered_boxes]}")
    
    # Categorize detections
    people_boxes, face_boxes, other_boxes = [], [], []
    for box in filtered_boxes:
        x1, y1, x2, y2, class_id, confidence = box
        detection = {
            'box': (x1, y1, x2, y2),
            'class_id': class_id,
            'confidence': confidence,
            'class_name': model.names[class_id]
        }
        
        if class_id in [0, 1]:  # Child or Adult
            people_boxes.append(detection)
        elif class_id in [2, 3]:  # Child or Adult face
            face_boxes.append(detection)
        else:
            other_boxes.append(detection)
    
    # Process face-person pairs
    for face in face_boxes:
        for person in people_boxes:
            if is_face_inside_person(face['box'], person['box']):
                scores = calculate_detection_scores(face, person, AP_values)
                update_detection_classes(face, person, scores, model.names)
                
                # Log scores for debugging
                for key, value in scores.items():
                    logging.debug(f"{key}: {value:.3f}")
    
    # Draw all detections
    detections = people_boxes + face_boxes + other_boxes
    for detection in detections:
        proximity = None
        if detection['class_id'] in [2, 3]:
            proximity = get_proximity(detection['box'], detection['class_name'])
        draw_detection(image, detection, proximity)
    
    # Save output
    output_dir = "/home/nele_pauline_suffo/outputs/yolo_all_detections"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    
    if cv2.imwrite(output_path, image):
        logging.info(f"Output saved at {output_path}")
    else:
        logging.error(f"Failed to save output image to {output_path}")
    
def main():
    parser = argparse.ArgumentParser(description="Run YOLO inference on an image")
    parser.add_argument("--image_path", type=str, help="Path to the image file")
    args = parser.parse_args()

    run_inference(args.image_path)
    
if __name__ == "__main__":
    main()
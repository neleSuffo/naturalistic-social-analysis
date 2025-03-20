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

def run_inference(image_path):
    # Load an image
    image = cv2.imread(image_path)
    results = model.predict(image, iou=YoloConfig.best_iou)
    filtered_boxes = custom_nms(results, iou_threshold=YoloConfig.between_classes_iou)

    # add logging about the detections
    logging.info(f"Detected classes after NMS: {[model.names[box[4]] for box in filtered_boxes]}")
    # Define Average Precision values
    AP_adult_face = 0.956
    AP_adult = 0.947
    AP_child_face = 0.856
    AP_child = 0.863
    
    # Lists to store people and face detections
    people_boxes = []
    face_boxes = []
    other_boxes = []

    # Separate filtered boxes into people and face detections
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

    for face in face_boxes:
        face_x1, face_y1, face_x2, face_y2 = face['box']
        face_class_id = face['class_id']
        face_confidence = face['confidence']

        for person in people_boxes:
            person_x1, person_y1, person_x2, person_y2 = person['box']
            person_class_id = person['class_id']
            person_confidence = person['confidence']

            # Check if face is completely inside person bounding box
            if (person_x1 <= face_x1 and person_y1 <= face_y1 and
                    person_x2 >= face_x2 and person_y2 >= face_y2):
                
                # Calculate all possible scores
                adult_face_score = AP_adult_face * face_confidence if face_class_id == 3 else 0
                adult_person_score = AP_adult * person_confidence if person_class_id == 1 else 0
                child_face_score = AP_child_face * face_confidence if face_class_id == 2 else 0
                child_person_score = AP_child * person_confidence if person_class_id == 0 else 0
                print(f"Adult face score: {adult_face_score}")
                print(f"Adult person score: {adult_person_score}")
                print(f"Child face score: {child_face_score}")
                print(f"Child person score: {child_person_score}")
                
                # First priority: If either detection is adult with high confidence
                if adult_face_score > 0 or adult_person_score > 0:
                    # Compare adult scores against child scores
                    adult_combined_score = adult_face_score + adult_person_score
                    child_combined_score = child_face_score + child_person_score
                    print(f"Adult combined score: {adult_combined_score}")
                    print(f"Child combined score: {child_face_score + child_person_score}")
                    if adult_combined_score > child_combined_score:
                        # Change both to adult class
                        face['class_id'] = 3  # Adult face
                        face['class_name'] = model.names[3]
                        person['class_id'] = 1  # Adult
                        person['class_name'] = model.names[1]
                    else:
                        # Change both to child class
                        face['class_id'] = 2  # Child face
                        face['class_name'] = model.names[2]
                        person['class_id'] = 0  # Child
                        person['class_name'] = model.names[0]
                else:
                    # If no adult detection, keep as child
                    face['class_id'] = 2  # Child face
                    face['class_name'] = model.names[2]
                    person['class_id'] = 0  # Child
                    person['class_name'] = model.names[0]
                    
    # Combine corrected detections for drawing
    detections = people_boxes + face_boxes + other_boxes

    # add logging about the detections
    logging.info(f"Detected classes after NMS and correction: {[detection['class_name'] for detection in detections]}")

    # Draw detections
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        class_id = detection['class_id']
        confidence = detection['confidence']
        class_name = detection['class_name']

        # Get proximity only for detected faces
        proximity = None
        if class_id in [2, 3]:
            bounding_box = [x1, y1, x2, y2]
            proximity = get_proximity(bounding_box, class_name)

        # Assign a unique color to each class
        color = CLASS_COLORS[class_id]

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)

        # Display class name and confidence score
        if proximity is not None:
            label = f"{class_name} {confidence:.2f}, Proximity: {proximity:.2f}"
        else:
            label = f"{class_name} {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, 0.5, 1)[0]
        text_x, text_y = x1, y1 - 5 if y1 - 5 > 10 else y1 + 15

        # Draw background rectangle for text
        cv2.rectangle(image, (text_x, text_y - text_size[1] - 2), (text_x + text_size[0], text_y + 2), color, -1)

        # Put the label text
        cv2.putText(image, label, (text_x, text_y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                            
    # Save the image with detections
    output_name = image_path.split("/")[-1]
    output_path = f"/home/nele_pauline_suffo/outputs/yolo_all_detections/{output_name}"
    cv2.imwrite(output_path, image)

    logging.info(f"Output saved at {output_path}")
    
def main():
    parser = argparse.ArgumentParser(description="Run YOLO inference on an image")
    parser.add_argument("--image_path", type=str, help="Path to the image file")
    args = parser.parse_args()

    run_inference(args.image_path)
    
if __name__ == "__main__":
    main()
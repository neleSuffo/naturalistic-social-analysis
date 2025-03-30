import cv2
import logging
import argparse
import os
from ultralytics import YOLO
from constants import YoloPaths
from config import YoloConfig

# Color mapping for visualization
CLASS_COLORS = {
    0: (215, 65, 117),    # Person
    1: (105, 22, 51),     # Face
    2: (199, 200, 126),   # Child body part
    5: (102, 120, 124),   # Book
    6: (141, 142, 61),    # Toy
    7: (45, 55, 58),      # Kitchenware
    8: (242, 192, 209),   # Screen
    9: (201, 210, 213),   # Food
    10: (217, 218, 169),  # Other object
}

def draw_detection(image, box, cls_id, conf, age_cls=None, gaze_cls=None, age_conf=None, gaze_conf=None):
    """Draw bounding box and label for a detection."""
    x1, y1, x2, y2 = map(int, box)
    color = CLASS_COLORS.get(cls_id, (255, 255, 255))
    
    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Create label based on detection type
    if cls_id == 0:  # Person
        age_text = 'Adult' if age_cls == 0 else 'Child'
        label = f"Person ({age_text} {age_conf:.2f}) {conf:.2f}"
    elif cls_id == 1:  # Face
        age_text = 'Adult' if age_cls == 0 else 'Child'
        gaze_text = "Gaze" if gaze_cls == 1 else "No Gaze"
        label = f"Face ({age_text} {age_conf:.2f}, {gaze_text} {gaze_conf:.2f}) {conf:.2f}"
    elif cls_id == 2:  # Child body part
        label = f"Body Part {conf:.2f}"
    else:  # Objects
        class_names = {
            5: "Book", 6: "Toy", 7: "Kitchenware", 
            8: "Screen", 9: "Food", 10: "Other"
        }
        class_name = class_names.get(cls_id, f"Class {cls_id}")
        label = f"{class_name} {conf:.2f}"
    
    # Draw label
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
    # Load models
    object_model = YOLO(YoloPaths.all_trained_weights_path)
    person_face_model = YOLO(YoloPaths.person_face_trained_weights_path)
    gaze_cls_model = YOLO(YoloPaths.gaze_trained_weights_path)
    face_cls_model = YOLO(YoloPaths.face_trained_weights_path)
    person_cls_model = YOLO(YoloPaths.person_trained_weights_path)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to read image: {image_path}")
        return
    
    # Create visualization copy
    vis_image = image.copy()
    
    # Run object detection model
    object_results = object_model(image)
    for r in object_results[0].boxes:
        if r.cls.item() in range(5, 11):  # Only process classes 5-10
            box = r.xyxy[0].cpu().numpy()
            cls_id = int(r.cls.item())
            conf = r.conf.item()
            draw_detection(vis_image, box, cls_id, conf)
    
    # Run person/face detection model
    person_face_results = person_face_model(image)
    for r in person_face_results[0].boxes:
        box = r.xyxy[0].cpu().numpy()
        cls_id = int(r.cls.item())
        conf = r.conf.item()
        
        # Extract ROI
        x1, y1, x2, y2 = map(int, box)
        roi = image[y1:y2, x1:x2]
        
        if cls_id == 0:  # Person
            # Run person age classification
            person_cls_results = person_cls_model(roi)
            person_age_cls = int(person_cls_results[0].probs.top1)
            person_age_conf = float(person_cls_results[0].probs.top1conf)
            draw_detection(vis_image, box, cls_id, conf, 
                         age_cls=person_age_cls, 
                         age_conf=person_age_conf)
            
        elif cls_id == 1:  # Face
            # Run face age and gaze classification
            face_cls_results = face_cls_model(roi)
            face_age_cls = int(face_cls_results[0].probs.top1)
            face_age_conf = float(face_cls_results[0].probs.top1conf)
            
            gaze_results = gaze_cls_model(roi)
            gaze_cls = int(gaze_results[0].probs.top1)
            gaze_conf = float(gaze_results[0].probs.top1conf)
            
            draw_detection(vis_image, box, cls_id, conf, 
                         age_cls=face_age_cls,
                         age_conf=face_age_conf,
                         gaze_cls=gaze_cls,
                         gaze_conf=gaze_conf)
            
        elif cls_id == 2:  # Child body part
            draw_detection(vis_image, box, cls_id, conf)
    
    # Save output
    output_dir = "/home/nele_pauline_suffo/outputs/detection_pipeline_results/inference_images"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    
    if cv2.imwrite(output_path, vis_image):
        logging.info(f"Output saved at {output_path}")
    else:
        logging.error(f"Failed to save output image to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run YOLO inference on an image")
    parser.add_argument("--image_path", type=str, required=True,
                       help="Path to the image file")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    run_inference(args.image_path)

if __name__ == "__main__":
    main()
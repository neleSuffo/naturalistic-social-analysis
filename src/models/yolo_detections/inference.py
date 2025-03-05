import cv2
import logging
import argparse
import random
from ultralytics import YOLO
from constants import YoloPaths
from estimate_proximity import get_proximity, describe_proximity

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

def run_inference(image_path):
    # Load an image
    image = cv2.imread(image_path)

    # Run inference
    results = model.predict(image)

    # Generate random colors for each class
    class_colors = {}

    # Draw detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers
            class_id = int(box.cls[0])  # Class ID
            confidence = float(box.conf[0])  # Confidence score
            class_name = model.names[class_id]  # Class name
            
            # get proximity for detected faces
            if class_id in [2,3]:  
                bounding_box = [x1, y1, x2, y2]
                proximity = get_proximity(bounding_box, class_name)
                #proximity_description = describe_proximity(proximity)

            # Assign a unique color to each class
            color = CLASS_COLORS[class_id]

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)

            # Display class name and confidence score
            if class_id in [2,3]:
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
from ultralytics import YOLO
import cv2
import random

# Load the YOLOv11 model
model = YOLO("/home/nele_pauline_suffo/models/yolo11_all_detection.pt")

# Load an image
image_path = "/home/nele_pauline_suffo/ProcessedData/quantex_videos_processed/quantex_at_home_id263986_2022_11_29_02/quantex_at_home_id263986_2022_11_29_02_002130.jpg"
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

        # Assign a unique color to each class
        if class_id not in class_colors:
            class_colors[class_id] = [random.randint(0, 255) for _ in range(3)]
        color = class_colors[class_id]

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Display class name and confidence score
        label = f"{class_name} {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, 0.5, 1)[0]
        text_x, text_y = x1, y1 - 5 if y1 - 5 > 10 else y1 + 15

        # Draw background rectangle for text
        cv2.rectangle(image, (text_x, text_y - text_size[1] - 2), (text_x + text_size[0], text_y + 2), color, -1)

        # Put the label text
        cv2.putText(image, label, (text_x, text_y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

# Save the image with detections
output_path = "/home/nele_pauline_suffo/outputs/yolo_all_detections/output.jpg"
cv2.imwrite(output_path, image)

print(f"Output saved at {output_path}")
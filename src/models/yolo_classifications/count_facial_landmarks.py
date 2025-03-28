import cv2
import dlib
from pathlib import Path
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

# Load Dlib's pre-trained face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/nele_pauline_suffo/models/shape_predictor_68_face_landmarks.dat")  # Ensure this file is in your working directory

def process_and_save_frame(frame_path, output_dir):
    """Process cropped face image to detect facial landmarks and save annotated frame."""
    image = cv2.imread(str(frame_path))
    if image is None:
        raise ValueError(f"Could not load image from {frame_path}")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create dlib rectangle for the entire image since it's already a cropped face
    height, width = gray_image.shape
    face_rect = dlib.rectangle(0, 0, width, height)
    
    # Get landmarks
    landmarks = predictor(gray_image, face_rect)
    
    # Draw and count landmarks
    feature_counts = {'left_eye': 0, 'right_eye': 0, 'nose': 0, 'mouth': 0}
    
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        
        if 36 <= n <= 41:  # Left eye
            color = (255, 0, 0)
            feature_counts['left_eye'] += 1
        elif 42 <= n <= 47:  # Right eye
            color = (0, 255, 0)
            feature_counts['right_eye'] += 1
        elif 27 <= n <= 35:  # Nose
            color = (0, 0, 255)
            feature_counts['nose'] += 1
        elif 48 <= n <= 67:  # Mouth
            color = (0, 255, 255)
            feature_counts['mouth'] += 1
        else:
            color = (128, 128, 128)
            
        cv2.circle(image, (x, y), 2, color, -1)

    # Log feature counts
    for feature, count in feature_counts.items():
        logging.info(f"{feature}: {count} landmarks detected")

    output_path = Path(output_dir) / f"landmarks_{Path(frame_path).name}"
    cv2.imwrite(str(output_path), image)
    logging.info(f"Annotated frame saved at {output_path}")

# Usage
image_path = Path("/home/nele_pauline_suffo/ProcessedData/quantex_gaze_input/quantex_at_home_id262565_2022_05_08_02_036240_face_0.jpg")
output_dir = Path("/home/nele_pauline_suffo/outputs/yolo_gaze_classification")
output_dir.mkdir(parents=True, exist_ok=True)

process_and_save_frame(image_path, output_dir)
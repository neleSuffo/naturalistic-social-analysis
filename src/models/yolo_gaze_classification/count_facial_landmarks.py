import cv2
import dlib
import os
import logging
import numpy as np

# Load pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/nele_pauline_suffo/models/shape_predictor_68_face_landmarks.dat")

LANDMARK_COUNTS = {
    'left_eye': 6,   # Points 36-41
    'right_eye': 6,  # Points 42-47 
    'nose': 9,       # Points 27-35
    'mouth': 20      # Points 48-67
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def estimate_gaze(cropped_face_image):
    frame = cropped_face_image.copy()
    if frame is None:
        raise ValueError("Invalid input image")
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize feature arrays and counts
    left_eye, right_eye, nose, mouth = [], [], [], []
    feature_counts = {
        'left_eye': 0,
        'right_eye': 0, 
        'nose': 0,
        'mouth': 0
    }

    try:
        # Detect face region first
        faces = detector(gray)
        if not faces:
            logging.warning("No face detected in image")
            return frame
            
        face = faces[0]  # Get first face
        landmarks = predictor(gray, face)  # Get landmarks for detected face region
        
        # Debug face detection
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 255, 255), 1)
        
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            
            if 36 <= i <= 41:  # Left eye
                left_eye.append((x, y))
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
                feature_counts['left_eye'] += 1
            elif 42 <= i <= 47:  # Right eye  
                right_eye.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                feature_counts['right_eye'] += 1
            elif 27 <= i <= 35:  # Nose
                nose.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                feature_counts['nose'] += 1
            elif 48 <= i <= 67:  # Mouth
                mouth.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                feature_counts['mouth'] += 1

        # Calculate visibility percentages
        for feature, count in feature_counts.items():
            percentage = (count / LANDMARK_COUNTS[feature]) * 100
            logging.info(f"{feature}: {count}/{LANDMARK_COUNTS[feature]} landmarks visible ({percentage:.1f}%)")
            
    except Exception as e:
        logging.error(f"Error detecting landmarks: {str(e)}")
        return frame

    return frame

# Apply gaze estimation on a single frame
image_path = "/home/nele_pauline_suffo/ProcessedData/quantex_gaze_input/quantex_at_home_id262691_2022_03_19_01_022770_face_0.jpg"
base_name = os.path.basename(image_path)
face_image = cv2.imread(image_path)
processed_frame = estimate_gaze(face_image)
output_dir = "/home/nele_pauline_suffo/outputs/yolo_gaze_classification"
output_path = os.path.join(output_dir, base_name)
logging.info("Saving processed frame to %s", output_path)
cv2.imwrite(output_path, processed_frame)
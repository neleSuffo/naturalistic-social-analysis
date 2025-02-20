import cv2
import dlib
import numpy as np

# Load pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/nele_pauline_suffo/models/shape_predictor_68_face_landmarks.dat")

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def get_eye_region(landmarks, start, end):
    points = [landmarks.part(i) for i in range(start, end)]
    region = np.array([(p.x, p.y) for p in points], dtype=np.int32)
    return region

def estimate_gaze_direction(eye_region, eye_image):
    # Create a mask for the eye region
    mask = np.zeros_like(eye_image)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(eye_image, mask)

    # Find the contours in the eye region
    contours, _ = cv2.findContours(eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour which will be the pupil
        contour = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(contour)
        cx, cy = x + w // 2, y + h // 2  # Center of the pupil

        # Determine the position of the pupil relative to the eye region
        if cx < eye_region[:, 0].mean() - 5:
            return "Looking left"
        elif cx > eye_region[:, 0].mean() + 5:
            return "Looking right"
        else:
            return "Looking center"
    return "Gaze undetermined"

def process_image(image_path):
    frame = cv2.imread(image_path)
    faces = detector(frame)
    
    for face in faces:
        # Get facial landmarks
        landmarks = predictor(frame, face)
        
        # Draw rectangle around face
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        
        # Draw facial landmarks
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
            
        # Draw specific features
        # Eyes (36-47)
        for n in range(36, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            
        # Nose (27-35)
        for n in range(27, 35):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

        # Get the left and right eye regions
        left_eye_region = get_eye_region(landmarks, 36, 42)
        right_eye_region = get_eye_region(landmarks, 42, 48)

        # Extract the eye images
        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])
        left_eye_image = gray[min_y:max_y, min_x:max_x]

        min_x = np.min(right_eye_region[:, 0])
        max_x = np.max(right_eye_region[:, 0])
        min_y = np.min(right_eye_region[:, 1])
        max_y = np.max(right_eye_region[:, 1])
        right_eye_image = gray[min_y:max_y, min_x:max_x]

        # Estimate gaze direction for both eyes
        left_gaze = estimate_gaze_direction(left_eye_region - [min_x, min_y], left_eye_image)
        right_gaze = estimate_gaze_direction(right_eye_region - [min_x, min_y], right_eye_image)

        # Determine overall gaze direction
        if left_gaze == "Looking center" and right_gaze == "Looking center":
            gaze_direction = "Looking at the camera"
        else:
            gaze_direction = "Looking away"

        # Display the result
        cv2.putText(frame, f"Gaze: {gaze_direction}", (face.left(), face.top() - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the image with gaze direction
    output_path = "/home/nele_pauline_suffo/ProcessedData/quantex_at_home_id254922_2022_04_19_01_000000_gaze.jpg"
    cv2.imwrite(output_path, frame)
    print(f"Processed image saved to: {output_path}")
    
# apply gaze estimation on a single frame
image_path = "/home/nele_pauline_suffo/ProcessedData/quantex_videos_processed/quantex_at_home_id254922_2022_04_19_01/quantex_at_home_id254922_2022_04_19_01_000000.jpg"
process_image(image_path)
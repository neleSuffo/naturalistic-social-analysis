import cv2
import torch
from retinaface import RetinaFace

def detect_faces(video_path):
    # Initialize RetinaFace detector
    detector = RetinaFace(quality="normal")

    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Iterate over frames in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB (RetinaFace expects RGB format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        faces = detector.predict(rgb_frame)

        # Draw bounding boxes around detected faces
        if faces is not None:
            for face in faces:
                x1, y1, x2, y2, _ = face
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Display the frame with detected faces
        cv2.imshow('Video', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Path to your video file
video_path = "path_to_your_video_file.mp4"

# Detect faces in the video using RetinaFace
detect_faces(video_path)
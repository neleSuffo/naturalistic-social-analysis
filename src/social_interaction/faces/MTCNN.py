import cv2
from facenet_pytorch import MTCNN

def detect_faces(video_path):
    # Initialize MTCNN face detector
    mtcnn = MTCNN()

    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Iterate over frames in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB (MTCNN expects RGB format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        boxes, _ = mtcnn.detect(rgb_frame)

        # Draw bounding boxes around detected faces
        if boxes is not None:
            for box in boxes:
                x, y, w, h = box.astype(int)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame with detected faces
        cv2.imshow('Video', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()
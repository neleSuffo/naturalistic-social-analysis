import cv2
import dlib

def detect_faces(video_path: str) -> None:
    """
    This function detects faces in a video file and draws bounding boxes around them.

    Parameters
    ----------
    video_path : str
        the path to the video file
    """
    # Initialize face detector from Dlib
    face_detector = dlib.get_frontal_face_detector()

    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Iterate over frames in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for face detection
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_detector(frame_gray)

        # Draw bounding boxes around detected faces
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame with detected faces
        cv2.imshow('Video', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()
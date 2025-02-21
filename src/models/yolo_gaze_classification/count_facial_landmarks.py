import cv2
import logging
import mediapipe as mp

logging.basicConfig(level=logging.INFO)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                  max_num_faces=1,
                                  refine_landmarks=True,  # Enables detection of refined landmarks around eyes and lips
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def process_and_save_frame(frame_path, output_dir):
    """
    Process the input frame to detect facial landmarks and save the annotated frame.

    Parameters:
    - frame_path: Path to the input image frame.
    - output_dir: Directory where the annotated frame will be saved.
    """
    # Read the image from the given path
    image = cv2.imread(frame_path)
    if image is None:
        print(f"Error reading image {frame_path}")
        return

    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    logging.info("Processing frame %s", frame_path)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw the face mesh annotations on the image.
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

    # Prepare the output path
    filename = os.path.basename(frame_path)
    output_path = os.path.join(output_dir, filename)
    logging.info("Saving processed frame to %s", output_path)

    # Save the annotated image to the output directory
    cv2.imwrite(output_path, image)


image_path = "/home/nele_pauline_suffo/ProcessedData/quantex_gaze_input/quantex_at_home_id262691_2022_03_19_01_022770_face_0.jpg"
output_dir = "/home/nele_pauline_suffo/outputs/yolo_gaze_classification"
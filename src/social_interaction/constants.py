class DetectionPaths:
    """
    The paths to the input and output folders for the detection models.
    """

    person: str = "output/output_person_detection/"
    face: str = "output/output_face_detection/"
    videos_input = "data/video/"
    results = "output/"


class DetectionParameters:
    """
    The parameters for the detection models.
    """

    # Every frame_step-th frame is processed
    frame_step = 30
    # The minimum length of a social interaction
    # Based on the frame step of 30 frames
    # Adjust accordingly so that the minimum length is at least 3 second
    interaction_length = 3
    # The class of the object to detect
    yolo_detection_class = "person"

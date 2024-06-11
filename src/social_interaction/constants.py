from collections import defaultdict


class DetectionPaths:
    """
    The paths to the input and output folders for the detection models.
    """

    person: str = "output/output_person_detection/"
    face: str = "output/output_face_detection/"
    videos_input = "data/video/"
    results = "output/"


class VTCParameters:
    # path variables for the voice-type-classifier
    audio_path = "data/audio/"
    environment_path = "/Users/nelesuffo/Library/Caches/pypoetry/virtualenvs/pyannote-afeazePz-py3.8/bin/python"
    execution_file_path = "src/social_interaction/language/run_vtc.py"
    execution_command = "/Users/nelesuffo/projects/voice_type_classifier/apply.sh"
    output_file_path = "output/output_voice_type_classifier/audio/all.rttm"
    df_output_path = "output/output_voice_type_classifier/"


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
    mtcnn_detection_class = "face"
    vtc_detection_class = "voice"


class LabelDictionaries:
    # Map labels to integers (defaultdict is used to return 99 for unknown labels)
    label_dict = defaultdict(
        lambda: 99,
        {
            "person": 1,
            "reflection": 2,
            "book": 3,
            "animal": 4,
            "toy": 5,
            "kitchenware": 6,
            "screen": 7,
            "food": 8,
            "object": 9,
            "other_object": 9,
            "face": 10,
            "voice": 20,
            "noise": -1,
        },
    )

    # Map label id to supercategory
    supercategory_dict = defaultdict(
        lambda: "unknown",
        {
            1: "person",
            2: "reflection",
            3: "object",
            4: "object",
            5: "object",
            6: "object",
            7: "object",
            8: "object",
            9: "object",
            10: "face",
            20: "voice",
            -1: "noise",
            99: "unknown",
        },
    )

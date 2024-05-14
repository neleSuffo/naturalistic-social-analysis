# Path variable to the input data
videos_input_path = "data/video/"

# Path variables for the person detection
video_person_output_path = "output/output_person_detection/"

# Path variables for the person detection
video_face_output_path = "output/output_face_detection/"

# Path to detection output json
detection_results_path = "output/"

# The frame step for the detection
# Every frame_step-th frame is processed
frame_step = 30

# The minimum length of a social interaction
# Based on the frame step of 30 frames
# Adjust accordingly so that the minimum length is at least 3 second
interaction_length = 3

import cv2
import my_utils
from pathlib import Path
import config
import logging
from timeit import default_timer as timer
import os

# Set up logging
logging.basicConfig(level=logging.INFO)


DETECTION_FUNCTIONS = {
    "person": my_utils.run_person_detection,
    "face": my_utils.run_frame_face_detection,
    "batch-wise face": my_utils.run_batch_face_detection,
    "voice": my_utils.run_voice_detection,
    "proximity": my_utils.run_proximity_detection,
}


MODELS = {
    "person": my_utils.load_person_detection_model,
    "face": my_utils.load_frame_face_detection_model,
    "batch-wise face": my_utils.load_batch_face_detection_model,
}


def run_detection(
    detection_type, detection_function, video_file, file_name, models, results
):
    logging.info(f"Performing {detection_type} detection...")
    if detection_type in ["person", "face"]:
        results[detection_type] = detection_function(
            video_file,
            file_name,
            models[detection_type],
            config.frame_step,
        )
    elif detection_type == "batch-wise face":
        results[detection_type] = detection_function(
            video_file,
            models[detection_type],
        )
    elif detection_type == "voice":
        logging.info("Performing voice detection...")
        if "person" in results:
            len_detection_list = len(results["person"])
        elif "face" in results:
            len_detection_list = len(results["face"])
        elif "batch-wise face" in results:
            len_detection_list = len(results["batch-wise face"])
        else:
            # Get the number of frames in the video
            cap = cv2.VideoCapture(video_file)
            len_detection_list = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_video_duration, voice_duration_sum, results["voice"] = detection_function(
            video_file, len_detection_list
        )
    elif detection_type == "proximity":
        logging.info("Performing proximity detection...")
        detection_function()


def run_detection_models(
    detections: dict,
) -> dict:
    """
    This function runs five different detection models:
    - Person detection
    - Face detection
    - Batch-wise face detection
    - Voice detection
    - Proximity detection
    For each detection model, the output is a list of 1s and 0s,
    where 1 indicates the presence of the object or event in the frame.

    Parameters
    ----------
    detections : dict
        a dictionary indicating which detection models to run
        (person, face, voice, proximity)

    Returns
    -------
    dict
        the results of each detection model
    file_name_short
        the name of the video file (without the extension)
    """
    # Load the person detection and face detection models
    models = {
        key: MODELS[key]()
        for key, value in detections.items()
        if value and key in MODELS
    }

    # Get a list of all video files in the folder
    video_files = [
        video_f
        for video_f in Path(config.videos_input_path).iterdir()
        if video_f.suffix.lower() == ".mp4"
    ]

    # Initialize the results dictionary
    results = {}

    # Process each video file
    for video_file in video_files:
        file_name = video_file.name
        file_name_short = video_file.stem
        video_file = str(video_file)
        logging.info(
            f"Starting social interactions detection pipeline for {file_name_short}..."
        )
        for detection_type, detection_function in DETECTION_FUNCTIONS.items():
            if detections[detection_type]:
                run_detection(
                    detection_type,
                    detection_function,
                    video_file,
                    file_name,
                    models,
                    results,
                )
    return results, file_name_short


def main(detections_dict: dict) -> None:
    """
    The main function of the social interactions detection pipeline.

    Parameters
    ----------
    detections_dict : dict
        a dictionary indicating which detection models to run
        (person, face, voice, proximity)

    """
    # Start the timer
    start_time = timer()

    # Run the detection models
    results, file_name_short = run_detection_models(detections_dict)

    # Save the results to a JSON file
    json_output_path = os.path.join(
        config.detection_results_path, f"{file_name_short}.json"
    )
    my_utils.save_results_to_json(results, json_output_path)

    # Create the final result list
    final_result = []
    # Sum the values for each frame
    for values in zip(*results.values()):
        final_result.append(sum(values))

    # Save the summed results to a JSON file
    json_output_path_summed = os.path.join(
        config.detection_results_path, f"{file_name_short}_summed.json"
    )
    my_utils.save_results_to_json(final_result, json_output_path_summed)

    # Count the sequences of 2 or 3 in final results
    sequence_nr_of_frames = my_utils.count_sequences(
        final_result, config.interaction_length
    )

    # Print the results
    my_utils.display_results(results)
    print(
        f"Number of consecutive frames with at least two detections in parallel: {sequence_nr_of_frames}"
    )

    # Stop the timer and print the runtime
    end_time = timer()
    runtime = end_time - start_time
    logging.info(f"Runtime: {runtime} seconds")


if __name__ == "__main__":
    detections_dict = {
        "person": True,
        "face": True,
        "batch-wise face": False,  # "batch-wise face" detection is faster than "face" detection
        "voice": True,
        "proximity": False,
    }
    main(detections_dict)

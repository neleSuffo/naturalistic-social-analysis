def calculate_percentage_and_print_results(
    detection_list: list, detection_type: str
) -> None:  # noqa: E501
    """
    This function calculates the percentage of frames
    where the object is detected and prints the results.

    Parameters
    ----------
    detection_list : list
        the list of detections
    detection_type : str
        the type of detection
    """
    percentage = sum(detection_list) / len(detection_list) * 100
    print(
        f"Percentages of at least one {detection_type} detected relative to the total frames: {percentage:.2f}"  # noqa: E231, E501
    )
    print(f"Total number of frames ({detection_type}): {len(detection_list)}")

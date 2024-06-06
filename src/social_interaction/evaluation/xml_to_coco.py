import xml.etree.ElementTree as ET
import config
import json
from collections import defaultdict

# This script reads the XML file containing the annotations and saves the data for each task to Parquet files.


def get_task_names_and_ids(root) -> dict:
    """
    This function extracts the task id and task name from the XML root.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        the root element of the XML file

    Returns
    -------
    dict
        a dictionary with task id as key and task name as value
    """
    task_names_and_ids_dict = {
        task.find("id").text: task.find("name").text for task in root.iter("task")
    }
    return task_names_and_ids_dict


def get_highest_frame_per_task(root: ET.Element, correction_value: int = 30) -> dict:
    """
    This function extracts the highest frame from each task in the XML root.

    Parameters
    ----------
    root : ET.Element
        the root element of the XML file
    correction_value : int
        the value by which the highest frame should be corrected

    Returns
    -------
    dict
        the extracted data in the form of a dictionary
        key: task id, value: highest frame
    """
    # Initialize an empty dictionary
    highest_frames_dict = defaultdict(int)

    # Iterate over all 'track' elements
    for track in root.findall(".//track"):
        task_id = track.get("task_id")

        # Iterate over all 'box' elements within the track
        for box in track.iter("box"):
            # Get the frame
            frame = int(box.get("frame"))
            # Update the highest frame for the task
            highest_frames_dict[task_id] = max(highest_frames_dict[task_id], frame)

    # Correct the highest frames by adding a correction value
    highest_frames_dict_corr = {
        key: value + correction_value for key, value in highest_frames_dict.items()
    }

    return highest_frames_dict_corr


def get_value_before_key(highest_frames_dict: dict, task_id: str) -> int:
    """
    This function gets the value before a specific key in the dictionary.

    Parameters
    ----------
    highest_frames_dict_corr : dict
        the dictionary from which to get the value
    task_id : str
        the key before which to get the value

    Returns
    -------
    int
        the value before the key
    """
    # Get the keys as a list
    keys = list(highest_frames_dict.keys())

    # Get the index of the key
    key_index = keys.index(task_id)

    # If the key is the first one or the last one, return 0 (no correction)
    if key_index == 0 or key_index == len(keys) - 1:
        return 0

    # Get the key before the key you have
    key_before = keys[key_index - 1]

    # Get the value from the key before the key you have
    value = highest_frames_dict[key_before]

    return value


def create_coco_annotation_format(root, task_details, highest_frames_dict) -> dict:
    """
    This function extracts data from the XML root and formats it in COCO format.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        the root element of the XML file
    task_details : dict
        the name and id per task extracted from the XML root
        key: task id, value: task name
    highest_frames_dict : dict
        the highest frame per task extracted from the XML root
        key: task id, value: highest frame

    Returns
    -------
    dict
        the extracted data
        key: task name, value: list of dictionaries
    """
    # Initialize an empty dictionary
    data = {
        "info": {
            "description": "COCO Video Dataset",
            "version": "1.0",
            "date_created": "2024-06-03",
        },
        "videos": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # Initialize an empty set for added categories and videos
    added_categories = set()
    added_videos = set()

    # Iterate over all 'track' elements
    for track in root.iter("track"):
        task_id = track.get("task_id")
        track_label = track.get("label")
        # Map the label to its corresponding label id using the dictionary
        # returns -1 if the label is not in the dictionary
        track_label_id = config.label_dict.get(track_label, -1)
        # Map the label to its corresponding supercategory using the dictionary
        supercategory = config.supercategory_dict.get(
            track_label_id, "unknown"
        )  # returns "unknown" if the label is not in the dictionary

        # Get the frame correction value
        frame_correction = get_value_before_key(highest_frames_dict, task_id)
        # Get the task name from the task_details dictionary
        task_name = task_details.get(task_id, next(iter(task_details.values())))

        # Add video details if not already added
        if task_id not in added_videos:
            data["videos"].append(
                {
                    "id": task_id,
                    "file_name": f"{task_name}.mp4",
                }
            )
            added_videos.add(task_id)

        # Add category details
        if track_label_id not in added_categories:
            data["categories"].append(
                {
                    "supercategory": supercategory,
                    "id": track_label_id,
                    "name": track_label,
                }
            )
            added_categories.add(track_label_id)

        # Iterate over each 'box' element within the 'track'
        for box in track.iter("box"):
            row = box.attrib
            row["track_label"] = track_label
            # Correct the frame value
            row["frame"] = int(row["frame"]) - frame_correction

            # Add to COCO format data
            data["images"].append(
                {
                    "id": len(data["images"]) + 1,
                    "video_id": task_id,
                    "frame_id": row["frame"],
                    "file_name": f"{task_name}_{row['frame']}.jpg",
                }
            )
            data["annotations"].append(
                {
                    "image_id": len(
                        data["images"]
                    ),  # the id of the image the annotation belongs to
                    "video_id": task_id,  # the id of the video the annotation belongs to
                    "category_id": track_label_id,  # the id of the category the annotation belongs to
                    "bbox": [
                        float(row["xtl"]),
                        float(row["ytl"]),
                        float(row["xbr"]),
                        float(row["ybr"]),
                    ],
                }
            )

    return data


def save_data_to_json(data: dict, base_path: str) -> None:
    """
    This function saves the data to a JSON file.

    Parameters
    ----------
    data : dict
        the extracted data in COCO format
    base_path : str
        the base path where the JSON file will be saved
    """
    with open(f"{base_path}/annotations.json", "w") as f:
        json.dump(data, f, indent=4)


# Parse XML
tree = ET.parse(config.annotations_input_path)
root = tree.getroot()

# Extract task names and ids
task_details = get_task_names_and_ids(root)

# Extract highest frame per task
highest_frame_dict = get_highest_frame_per_task(root)
highest_frame_dict_sorted = {
    key: highest_frame_dict.get(key, 0) for key in task_details.keys()
}

data = create_coco_annotation_format(root, task_details, highest_frame_dict_sorted)
save_data_to_json(data, config.annotations_output_path)

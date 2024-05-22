import pandas as pd
import xml.etree.ElementTree as ET
import config


def get_task_details(root) -> dict:
    """
    This function extracts task details from the XML root.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        the root element of the XML file

    Returns
    -------
    dict
        a dictionary with task id as key and task name as value
    """
    return {task.find("id").text: task.find("name").text for task in root.iter("task")}


def get_data(root, task_details):
    """
    This function extracts data from the XML root.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        the root element of the XML file
    task_details : dict
        the task details extracted from the XML root
        key: task id, value: task name

    Returns
    -------
    dict
        the extracted data
        key: task name, value: list of dictionaries
    """
    data = {}
    for track in root.iter("track"):
        # Get the task label and task id
        track_label = track.get("label")
        task_id = track.get("task_id")
        # Get the task name from the task_details dictionary
        task_name = (
            task_details[task_id]
            if task_id is not None
            else next(iter(task_details.values()))
        )
        data.setdefault(task_name, [])
        # Iterate over each 'box' element within the 'track'
        for box in track.iter("box"):
            row = box.attrib
            row["track_label"] = track_label
            data[task_name].append(row)
    return data


def save_data_to_parquet(data: dict, base_path: str) -> None:
    """
    This function saves the data to Parquet files.

    Parameters
    ----------
    data : dict
        the extracted data
        key: task name, value: list of dictionaries
    base_path : str
        the base path where the Parquet files will be saved
    """
    for task_id, rows in data.items():
        df = pd.DataFrame(rows)
        df.to_parquet(f"{base_path}/{task_id}_annotation.parquet")


# Parse XML
tree = ET.parse(config.annotations_input_path)
root = tree.getroot()

# Extract data and save to Parquet
task_details = get_task_details(root)
data = get_data(root, task_details)
save_data_to_parquet(data, config.annotations_output_path)

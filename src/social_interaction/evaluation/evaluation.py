import pandas as pd
import config
import json
from pathlib import Path
from preprocessing import preprocess_annotations
import my_utils


def prepare_labels():
    """
    This function preprocesses the annotations xml files in the annotations folder.
    It returns two dictionaries with dataframes containing the labels as boolean values.
    The key is the name of the xml file.

    Returns
    -------
    annotation_dict: dict
        a dictionary containing the dataframes
        with annotations with the labels as boolean values and bounding boxes from all xml files
        the key is the name of the xml file
    label_bool_dict: dict
        a dictionary containing the dataframes
        with the annotations (only labels as boolean values) from all xml files
        the key is the name of the xml file
    """
    # create a dataframe for each xml file in the folder
    directory = Path(config.annotations_path)

    # Get a list of all XML files in the directory
    xml_files = list(directory.glob("*.xml"))

    annotation_dict = {}
    label_bool_dict = {}

    # Iterate over the XML files
    # This is only executed once to process all xml files in the directory
    # and save the dataframes to Parquet files
    for xml_file in xml_files:
        filename = xml_file.stem  # Get the filename without the extension

        annotation_df_rows, label_bool_df = preprocess_annotations(directory, xml_file)
        annotation_dict[filename] = annotation_df_rows
        label_bool_dict[filename] = label_bool_df

    return annotation_dict, label_bool_dict


def prepare_person_output() -> dict:
    """
    This function reads the output json file and returns
    a dictionary with dataframes containing the output for person detection
    as boolean values. The key is the name of the json file.

    Returns
    -------
    output_dict: dict
        the dictionary containing the dataframes with the output as boolean values
        the key is the name of the json file
    """
    # create a dataframe for each json file in the folder
    directory = Path(config.model_output_path)

    # Get a list of all json files in the directory
    json_files = list(directory.glob("*.json"))

    # Iterate over the json files
    # This is only executed once to process all json files in the directory
    output_dict = {
        file.stem: pd.DataFrame(
            json.load(file.open()).get("person"), columns=["person"]
        ).astype(bool)
        for file in json_files
    }

    for filename, output_df in output_dict.items():
        # Add the index as a column
        # modify the index to be the frame number
        # TODO: find a way to import config.frame_step
        output_df["frame"] = output_df.index * 30 + 30

        # Reorder the columns
        output_df = output_df[["frame", "person"]]

        # Update the DataFrame in output_dict
        output_dict[filename] = output_df

    return output_dict


def main():
    """
    Main function to evaluate the model output.
    """
    # Get the dataframes with the labels as boolean values
    annotation_dict_big, annotation_dict = prepare_labels()

    # Get the dataframes with the person output as boolean values
    output_dict = prepare_person_output()

    # Combine both dictionaries
    combined_dict = {}

    for key in annotation_dict.keys():
        if key in output_dict.keys():
            combined_df = annotation_dict[key].merge(
                output_dict[key], how="left", left_on="frame", right_on="frame"
            )
            # Some frames are labeled as "Noise", those frames are not considered in the annotations
            # They will be labeled as False, as nothing could be detected
            combined_df = combined_df.fillna(False)
            combined_dict[key] = combined_df

    # Combine the dataframes from all files to one big dataframe
    evaluation_df = pd.concat(combined_dict.values(), ignore_index=True)

    # Some annotations don't have all label categories (show as Nan)
    # Convert them to False
    evaluation_df = evaluation_df.fillna(False)

    # Calculate accuracy and confusion matrix
    my_utils.calculate_accuracy(evaluation_df)
    my_utils.generate_confusion_matrix(evaluation_df)


if __name__ == "__main__":
    main()

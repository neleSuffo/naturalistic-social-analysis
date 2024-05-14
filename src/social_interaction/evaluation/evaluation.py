import pandas as pd
import config
import glob
import json
import os
from preprocessing import preprocess_annotations


def prepare_labels():
    """
    This function preprocesses the annotations xml files in the annotations folder.
    It returns the dataframes with the labels as boolean values.

    Returns
    -------
    _type_
        _description_
    """
    # create a dataframe for each xml file in the folder
    directory = config.annotations_path

    # Get a list of all XML files in the directory
    xml_files = glob.glob(os.path.join(directory, "*.xml"))

    # Iterate over the XML files
    # This is only executed once to process all xml files in the directory
    # and save the dataframes to Parquet files
    for xml_file in xml_files:
        annotation_df_rows, label_bool_df = preprocess_annotations(directory, xml_file)
    return annotation_df_rows, label_bool_df


def prepare_model_output():
    """
    This function reads the output json file and returns a dataframe with the output as boolean values.

    Returns
    -------
    _type_
        _description_
    """
    # Read the output JSON file
    with open("output/257608_005.json", "r") as file:
        output_dict = json.load(file)

    # just for now
    output_dict["voice"] = output_dict["voice"][:-1]

    # Convert the output dictionary to a pandas DataFrame
    # and cast the values to boolean
    output_df = pd.DataFrame(output_dict).astype(bool)

    # Add the index as a column
    # modify the index to be the frame number
    # TODO: find a way to import config.frame_step
    output_df = output_df.reset_index()
    output_df["frame"] = output_df["index"] * 30 + 30

    # Drop the index column
    output_df.drop(columns=["index"], inplace=True)

    # Reorder the columns
    output_df = output_df[["frame", "person", "face", "voice"]]

    return output_df


if __name__ == "__main__":
    annotation_df_rows, label_bool_df = prepare_labels()
    output_df = prepare_model_output()

# Combine the dataframes
combined_df = output_df.merge(
    label_bool_df, how="left", left_on="frame", right_on="frame"
)

# Some frames are labeled as "Noise", those frames are not considered in the annotations
# They will be labeled as False, as nothing could be detected
combined_df[label_bool_df.columns] = combined_df[label_bool_df.columns].fillna(False)

# Create accuracy column: True if person is True and at least one label (person or reflection) is True,
# or if person is False and both labels (person and reflection) are False as well
combined_df["accuracy"] = (
    (combined_df["person"]) & (combined_df["label_0"] | combined_df["label_1"])
) | ((~combined_df["person"]) & (~combined_df["label_0"] & ~combined_df["label_1"]))

# Calculate the number of rows
num_rows = combined_df.shape[0]

# Calculate the true counts for every column
true_counts = combined_df.sum()

print(f"Accuracy is: {true_counts.iloc[-1]/num_rows*100:.2f}%")

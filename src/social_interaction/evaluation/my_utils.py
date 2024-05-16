import pandas as pd
import xml.etree.ElementTree as ET
import config
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def generate_confusion_matrix(df: pd.DataFrame) -> None:
    """
    Generate and plot the confusion matrix.

    Parameters
    ----------
    df : pd.DataFrame
        the DataFrame containing the labels and annotations
    """
    # Convert the columns to the appropriate data types
    df[["person", "label_0", "label_1"]] = df[["person", "label_0", "label_1"]].astype(
        int
    )

    # Combine the labels
    df["predicted"] = df["label_0"] | df["label_1"]

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(df["predicted"], df["person"])
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()
    plt.show()


def calculate_accuracy(df: pd.DataFrame) -> None:
    """
    This function calculates and prints the accuracy.

    Parameters
    ----------
    df : pd.DataFrame
        the DataFrame containing the labels and annotations
    """
    # Calculate the number of rows
    num_rows = df.shape[0]

    # Calculate accuracy
    accuracy = (
        (
            ((df["person"]) & (df["label_0"] | df["label_1"]))
            | ((~df["person"]) & (~df["label_0"] & ~df["label_1"]))
        ).sum()
        / num_rows
        * 100
    )
    print(f"Accuracy is: {accuracy:.2f}%")


def pivot_df(df: pd.DataFrame, dummies_label_list: list) -> pd.DataFrame:
    """
    This function pivots the DataFrame and sorts it by frame.
    The resulting DataFrame has a row for each frame with the labels as columns

    Parameters
    ----------
    df : pd.DataFrame
        the DataFrame containing the data from the XML file

    dummies_label_list : list
        the list of the labels existing in the DataFrame

    Returns
    -------
    pd.DataFrame
        the DataFrame containing the data from the XML file
        pivoted and sorted by frame
        (now each frame has a row with the labels as columns and True/False values for each label)
    """

    # Pivot the table
    pivot_df = df.pivot_table(
        index=["frame", "outside", "keyframe"] + dummies_label_list,
        columns="label",
        aggfunc="first",
    )
    # Reset the index
    pivot_df.reset_index(inplace=True)

    # Flatten the MultiIndex in columns
    pivot_df.columns = [
        "_".join(map(str, col)).strip() for col in pivot_df.columns.values
    ]

    # Sort dataframe by frame
    pivot_df.sort_values(by="frame_", ascending=True, inplace=True)

    return pivot_df


def process_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    This function processes the labels in the DataFrame.
    It takes into account that 'ID', 'Age', and 'Visibility' only contain
    values for the 'Person' and 'Reflection' labels, and 'Interaction' only
    contains values for the 'book', 'animal', 'toy', 'kitchenware', and 'screen' labels.
    It creates dummy variables for each label and concatenates them with the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        the DataFrame containing the data from the XML file

    Returns
    -------
    df: pd.DataFrame
        the DataFrame containing the data from the XML file
        concatenated with the dummy variables for each label
    label_bool_df: pd.DataFrame
        the DataFrame containing the dummy variables for each label
        (True/False) for each frame
    list
        a list of the labels existing in the DataFrame
        (some annotations may not have all labels)
    """

    # Set 'ID', 'Age', and 'Visibility' to NaN where 'label' is not 'Person' or 'Reflection'
    df.loc[~df["label"].isin([0, 1]), ["ID", "Age", "Visibility"]] = np.nan

    # Set 'Interaction' to NaN where 'label' is not 'book', 'animal', 'toy', 'kitchenware', 'screen'
    df.loc[~df["label"].isin([2, 3, 4, 5, 6]), "Interaction"] = np.nan

    # Create dummy variables for each label
    label_columns_df = pd.get_dummies(df[["frame", "label"]], columns=["label"])
    # Combine rows with the same frame so that the labels are combined
    label_bool_df = label_columns_df.groupby("frame").any().reset_index()

    # Get list of existing labels in the DataFrame
    label_list = sorted(df["label"].unique().tolist())

    # Concatenate dummies with the reduced annotation dataframe
    dummies = pd.get_dummies(df["label"])
    df = pd.concat([df, dummies], axis=1)

    return df, label_bool_df, label_list


def process_xml_file(file_path: str) -> pd.DataFrame:
    """
    This function processes an XML file and returns a DataFrame.
    It converts the columns to the appropriate data types and
    replaces NaN values with -1.

    Parameters
    ----------
    file_path : str
        the path to the XML file

    Returns
    -------
    pd.DataFrame
        the DataFrame containing the data from the XML file
    """
    # Apply the xml_to_df function to the file
    df = xml_to_df(file_path)

    # Drop unimportant columns
    df.drop(columns=config.drop_columns, inplace=True)

    # Get the columns, create them if they don't exist
    # (to catch errors because of non-existing columns)
    df["Interaction"] = df.get("Interaction", pd.Series([""] * len(df)))
    df["Age"] = df.get("Age", pd.Series([""] * len(df)))
    df["Visibility"] = df.get("Visibility", pd.Series([""] * len(df)))
    df["ID"] = df.get("ID", pd.Series([""] * len(df)))

    # Replace 'yes' and 'no' values with 1 and 0 for the 'Interaction' column
    df["Interaction"] = df["Interaction"].str.lower().map(config.interaction_map)

    # Map str values to int values for the 'Age' and 'label' column
    df["Age"] = df["Age"].str.lower().map(config.age_map)
    df["label"] = df["label"].str.lower().map(config.label_map)

    # Convert the column to numeric type, coercing non-numeric values to NaN
    df[config.int_columns] = df[config.int_columns].apply(
        lambda x: pd.to_numeric(x, errors="coerce")
    )

    # Replace NaN values for visibility and ID with -1
    df[config.int_columns] = df[config.int_columns].fillna(-1)

    # Convert the columns to the appropriate data types
    df[config.int_columns] = df[config.int_columns].astype(int)
    df[config.float_columns] = df[config.float_columns].astype(float)

    # Sort the DataFrame by frame
    df.sort_values(by="frame", inplace=True)

    return df


def xml_to_df(file_path):
    """
    This function reads an XML file and converts it to a DataFrame.

    Parameters
    ----------
    file_path : str
        the path to the XML file
    """
    # Parse the XML file
    tree = ET.parse(file_path)

    # Get the root element
    root = tree.getroot()

    # Create a list to hold the data
    data = []

    # Iterate over each 'track' element in root
    for track in root.iter("track"):
        # Get the attributes of the 'track' element
        track_data = track.attrib

        # Iterate over each 'box' element within the 'track'
        for box in track.iter("box"):
            # Create a dictionary to hold the data for this box
            box_data = box.attrib

            # Get the 'attribute' element
            attribute = box.find("attribute")
            if attribute:
                # Add the attribute name and value to box_data
                box_data[attribute.attrib["name"]] = attribute.text

            # Combine track_data and box_data
            combined_data = {**track_data, **box_data}

            # Add combined_data to data
            data.append(combined_data)

    # Create a DataFrame from data
    df = pd.DataFrame(data)

    return df

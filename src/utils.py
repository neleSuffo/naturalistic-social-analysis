import pandas as pd
import xml.etree.ElementTree as ET


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

            # Add the attribute name and value to box_data
            box_data[attribute.attrib["name"]] = attribute.text

            # Combine track_data and box_data
            combined_data = {**track_data, **box_data}

            # Add combined_data to data
            data.append(combined_data)

    # Create a DataFrame from data
    df = pd.DataFrame(data)

    return df

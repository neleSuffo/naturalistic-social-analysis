import xml.etree.ElementTree as ET
import logging

# Configure logging
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


def create_file_name_id_dict(input_path: str) -> dict:
    """
    The function creates a dictionary 
    with the file name as the key and the file id as the value.

    Parameters
    ----------
    xml_data : str
        the path to the XML file

    Returns
    -------
    dict
        the dictionary with 
        file name as key and file id as value
    """
    # Create a new dictionary to store the results
    result_dict = {}
    
    try:
        # Parse XML
        tree = ET.parse(input_path)
        root = tree.getroot()
        
        # Iterate over the 'task' elements
        for task in root.iter('task'):
            # Get the 'name' and 'id' sub-elements
            name = task.find('name').text
            id = task.find('id').text

            # Add the 'name' and 'id' to the dictionary
            result_dict[name] = id

    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
        
    return result_dict
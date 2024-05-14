import os
import my_utils
import config
import glob

# create a dataframe for each xml file in the folder
directory = config.annotations_path

# Get a list of all XML files in the directory
xml_files = glob.glob(os.path.join(directory, "*.xml"))

# Iterate over the XML files
for xml_file in xml_files:
    # Apply the xml_to_df function to the file
    annotation_df_columns = my_utils.process_xml_file(xml_file)

    # Process the labels
    annotation_df_columns, reduced_annotation_df, dummies_label_list = (
        my_utils.process_labels(annotation_df_columns)
    )

    # Pivot the table
    annotation_df_rows = my_utils.pivot_df(annotation_df_columns, dummies_label_list)

    # Save df to Parquet file
    base_name = os.path.splitext(os.path.basename(xml_file))[0]
    output_path = os.path.join(directory, f"{base_name}.parquet")
    annotation_df_rows.to_parquet(output_path)

import my_utils


def preprocess_annotations(directory: str, xml_file: str):
    """
    This function preprocesses the annotations xml files in the annotations folder.
    It creates a dataframe for each xml file in the folder,
    processes the labels, pivots the table and saves the df to a Parquet file.

    Parameters
    ----------
    directory: str
        the path to the directory containing the XML files
    xml_file: str
        the path to the XML file

    Returns
    -------
    annotation_df_rows: pd.DataFrame
        the DataFrame containing the data from the XML file
        pivoted and sorted by frame
        (now each frame has a row with the labels as columns and True/False values for each label)
    reduced_annotation_df
        the DataFrame containing the label information (True/False values for each label)
        for every frame

    """
    # Apply the xml_to_df function to the file
    annotation_df_columns = my_utils.process_xml_file(xml_file)

    # Process the labels
    annotation_df_columns, label_bool_df, dummies_label_list = my_utils.process_labels(
        annotation_df_columns
    )

    # Pivot the table
    annotation_df_rows = my_utils.pivot_df(annotation_df_columns, dummies_label_list)

    # Convert all column names to strings
    annotation_df_rows.columns = annotation_df_rows.columns.astype(str)
    label_bool_df.columns = label_bool_df.columns.astype(str)

    # Save df to Parquet file
    # base_name = os.path.splitext(os.path.basename(xml_file))[0]
    # output_path_full_label_df = os.path.join(directory, f"{base_name}.parquet")
    # output_path_bool_label_df = os.path.join(directory, f"{base_name}_bool.parquet")
    # annotation_df_rows.to_parquet(output_path_full_label_df)
    # label_bool_df.to_parquet(output_path_bool_label_df)

    return annotation_df_rows, label_bool_df

import argparse
import os
import json
from pathlib import Path
from typing import Dict
import pandas as pd
import logging
from my_utils import rttm_to_dataframe, process_all_json_files_to_dataframe
from constants import VTCPaths, DetectionPaths
from config import AnnotationProcessing
from pyannote.metrics.detection import DetectionPrecisionRecallFMeasure
from pyannote.core import Annotation, Segment

def main(dataset_type: str) -> None:
    """
    This function processes the VTC output for a given dataset and evaluates the performance for the childlens dataset.
    
    Parameters:
    ----------
    dataset_type: str
        Type of dataset: either 'childlens' or 'quantex'.
    """
    if dataset_type == "quantex":
        output_path = VTCPaths.quantex_df_file_path
        folder_path = VTCPaths.quantex_output_folder
    elif dataset_type == "childlens":
        output_path  = VTCPaths.childlens_df_file_path
        folder_path = VTCPaths.childlens_output_folder
        
    # convert vtc output to df
    rttm_file = os.path.join(folder_path, "all.rttm")
    rttm_to_dataframe(rttm_file, output_path)
    
    if dataset_type == "childlens":
        base_input_dir = DetectionPaths.childlens_annotations_dir
        # convert ground truth annotations to df
        for subfolder in base_input_dir.glob("annotations_*"):
            if subfolder.is_dir():
                logging.info(f"Processing subfolder: {subfolder}")
                
                # Process all JSON files and convert to DataFrame
                combined_df = process_all_json_files_to_dataframe(subfolder)
                
                # Generate the output file path based on the subfolder name
                output_file = base_input_dir / f"{dataset_type}_{subfolder.name}"
                
                # Save the DataFrame to a pickle file
                combined_df.to_pickle(output_file.with_suffix('.pkl'))
                logging.info(f"DataFrame saved to: {output_file.with_suffix('.pkl')}")

        # Evaluate the performance of the VTC output
        reference_df = pd.read_pickle(VtcPaths.childlens_gt_df_file_path)
        hypothesis_df = pd.read_pickle(VtcPaths.childlens_df_file_path)
        reference = dataframe_to_annotation(reference_df)
        hypothesis = dataframe_to_annotation(hypothesis_df)

        # Initialize metric
        detection_metric = DetectionPrecisionRecallFMeasure(collar=0, skip_overlap=False)

        # Compute precision, recall, and F1 score
        precision, recall, f1 = detection_metric(reference, hypothesis)

        # Print results
        logging.info(f"Precision: {precision:.2f}")
        logging.info(f"Recall: {recall:.2f}")
        logging.info(f"F1 Score: {f1:.2f}")
        
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Postprocess VTC output by converting all.rttm to a DataFrame."
    )
    parser.add_argument(
        "dataset_type",
        type=str,
        choices=["childlens", "quantex"],
        help="Type of dataset: either 'childlens' or 'quantex'."
    )
    args = parser.parse_args()
    
    main(args.dataset_type)
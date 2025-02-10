import argparse
import os
from my_utils import rttm_to_dataframe
from constants import VTCPaths

def main():
    parser = argparse.ArgumentParser(
        description="Postprocess VTC output by converting all.rttm to a DataFrame."
    )
    parser.add_argument(
        "folder_path",
        type=str,
        help="Folder path containing the all.rttm file."
    )
    parser.add_argument(
        "dataset_type",
        type=str,
        choices=["childlens", "quantex"],
        help="Type of dataset: either 'childlens' or 'quantex'."
    )
    args = parser.parse_args()
    
    if args.dataset_type == "childlens":
        output_path = VTCPaths.childlens_df_file_path
    else:
        output_path = VTCPaths.quantex_df_file_path

    rttm_file = os.path.join(args.folder_path, "all.rttm")
    rttm_to_dataframe(rttm_file, output_path)

if __name__ == "__main__":
    main()
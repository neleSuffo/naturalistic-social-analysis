import os
import argparse
from prepare_data.process_annotations.create_database import (
    write_xml_to_database,
    create_db_table_video_name_id_mapping,
    correct_erronous_videos_in_db,
    create_child_class_in_db,
)
from prepare_data.process_annotations.create_file_name_id_dict import create_file_name_id_dict
from prepare_data.process_annotations.convert_to_yolo import main as convert_to_yolo
from prepare_data.process_annotations.convert_to_mtcnn import main as convert_to_mtcnn
from constants import DetectionPaths

def main(model: str,
         yolo_target: str, 
         setup_db: bool = False) -> None:
    """ Main function to process annotations. The function creates a database and converts the annotations to YOLO and MTCNN format.
    
    Parameters
    ----------
    model : str
        Model to convert to (e.g., "yolo", "mtcnn", "all")
    yolo_target : str
        Target YOLO label (e.g., "face")
    setup_db : bool
        Whether to set up the database
    
    Returns
    -------
    None
    
    """
    os.environ['OMP_NUM_THREADS'] = '10'
    if setup_db:
        # Create a dictionary with the task name as key and a file id as value
        task_file_id_dict = create_file_name_id_dict(DetectionPaths.file_name_id_mapping_path)
        # Create a database table for the video file names and ids
        create_db_table_video_name_id_mapping(task_file_id_dict)
        # Convert the XML annotations to COCO format and store the results in a database
        write_xml_to_database()
        # Delete the erroneous videos in the database and add the new data from the individual xml files
        correct_erronous_videos_in_db()
        
        # Exclude child body parts from the class "person" in the YOLO labels
        # Create new class "child_body_parts" and update database
        create_child_class_in_db()
    
    # Convert the annotations to YOLO format and MTCNN format
    if model == 'yolo':
        convert_to_yolo(yolo_target)
    if model == 'mtcnn':
        convert_to_mtcnn()
    if model == 'all':
        convert_to_yolo(yolo_target)
        convert_to_mtcnn()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process annotations")
    parser.add_argument('model_target', type=str, help='Model to convert to (e.g., "yolo", "mtcnn", "all")')
    parser.add_argument('yolo_target', type=str, help='Target YOLO label (e.g., "face")')
    parser.add_argument('--setup_db', action='store_true', help='Whether to set up the database')

    args = parser.parse_args()
    main(model=args.model_target, yolo_target=args.yolo_target, setup_db=args.setup_db)
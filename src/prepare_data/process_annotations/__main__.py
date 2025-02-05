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

def main(model: str, yolo_target: str, setup_db: bool = False) -> None:
    """
    Main function to process annotations. This function creates a database 
    and converts annotations to YOLO and/or MTCNN format.
    
    Parameters
    ----------
    model : str
        Model to convert to (e.g., "yolo", "mtcnn", "all")
    yolo_target : str
        Target YOLO label ("person", "face" or "gaze")
    setup_db : bool
        Whether to set up the database

    Returns
    -------
    None
    """
    # Validate model argument
    if model not in {"yolo", "mtcnn", "all"}:
        raise ValueError(f"Invalid model '{model}'. Choose 'yolo', 'mtcnn', or 'all'.")
    
    os.environ['OMP_NUM_THREADS'] = '10'
    if setup_db:
        # Database setup
        task_file_id_dict = create_file_name_id_dict(DetectionPaths.file_name_id_mapping_path)
        create_db_table_video_name_id_mapping(task_file_id_dict)
        write_xml_to_database()
        correct_erronous_videos_in_db()
        create_child_class_in_db()

    # Annotation conversion
    if model == "yolo":
        convert_to_yolo(yolo_target)
    elif model == "mtcnn":
        convert_to_mtcnn()
    elif model == "all":
        convert_to_yolo(yolo_target)
        convert_to_mtcnn()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process annotations")
    parser.add_argument('model_target', type=str, help='Model to convert to (e.g., "yolo", "mtcnn", "all")')
    parser.add_argument('yolo_target', type=str, help='Target YOLO label ("person", "face" or "gaze")')
    parser.add_argument('--setup_db', action='store_true', help='Whether to set up the database')

    args = parser.parse_args()
    main(model=args.model_target, yolo_target=args.yolo_target, setup_db=args.setup_db)
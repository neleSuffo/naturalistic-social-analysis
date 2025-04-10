import os
import argparse
from prepare_data.process_annotations.create_database import (
    write_xml_to_database,
    create_child_class_in_db,
)
from prepare_data.process_annotations.convert_to_yolo import main as convert_to_yolo
from constants import DetectionPaths

def main(model: str = None, yolo_target: str = None, setup_db: bool = False) -> None:
    """
    Main function to process annotations. This function creates a database 
    and converts annotations to YOLO format.
    
    Parameters
    ----------
    model : str, optional
        Model to convert to (e.g., "yolo"). Defaults to None.
    yolo_target : str, optional
        Target YOLO label ("person_face", "person_face_object" or "gaze"). Defaults to None.
    setup_db : bool
        Whether to set up the database
    """
    # Validate arguments
    if model is not None:
        valid_models = {"yolo", "all"}
        valid_targets = {"all", "adult_person_face", "child_person_face", "person_face", 
                        "person_face_object", "object", "person", "face", "gaze"}
        
        if model not in valid_models:
            raise ValueError(f"Invalid model '{model}'. Must be one of: {valid_models}")
        
        if yolo_target not in valid_targets:
            raise ValueError(f"Invalid yolo_target '{yolo_target}'. Must be one of: {valid_targets}")
    
    try:
        os.environ['OMP_NUM_THREADS'] = '20'
        if setup_db:
            write_xml_to_database()
            create_child_class_in_db()

        if model in ["yolo", "all"]:
            convert_to_yolo(yolo_target)
            
    except Exception as e:
        print(f"Error processing annotations: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process annotations")
    parser.add_argument('--model_target', type=str, default=None, 
                       help='Model to convert to (e.g., "yolo", "all")')
    parser.add_argument('--yolo_target', type=str, default=None,
                       help='Target YOLO label ("all", "adult_person_face" or "gaze")')
    parser.add_argument('--setup_db', action='store_true', 
                       help='Whether to set up the database')

    args = parser.parse_args()
    main(model=args.model_target, yolo_target=args.yolo_target, setup_db=args.setup_db)
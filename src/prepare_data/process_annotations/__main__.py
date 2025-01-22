import os
from prepare_data.process_annotations.create_database import write_xml_to_database, create_db_table_video_name_id_mapping, correct_erronous_videos_in_db, create_child_class_in_db
from prepare_data.process_annotations.create_file_name_id_dict import create_file_name_id_dict
from prepare_data.process_annotations.convert_to_yolo import main as convert_to_yolo
from prepare_data.process_annotations.convert_to_mtcnn import main as convert_to_mtcnn
from constants import DetectionPaths

def main(yolo_target: str, setup_db: bool = False):
    os.environ['OMP_NUM_THREADS'] = '10'
    if setup_db == True:
        # Create a dictionary with the task name as key and a file id as valuey
        task_file_id_dict = create_file_name_id_dict(DetectionPaths.file_name_id_mapping_path)
        # Create a database table for the video file names and ids
        create_db_table_video_name_id_mapping(task_file_id_dict)
        # Convert the XML annotations to COCO format and store the results in a database
        write_xml_to_database()
        # Delete the erroneous videos in the database and add the new data form the indivual xml files
        correct_erronous_videos_in_db()
        
        # Exclude child body parts from the class "person" in the YOLO labels
        # Create new class "child_body_parts" and update database
        create_child_class_in_db()
    
    # Convert the annotations to YOLO format and MTCNN format
    #convert_to_yolo(yolo_target)
    convert_to_mtcnn()


if __name__ == "__main__":
    main('face')

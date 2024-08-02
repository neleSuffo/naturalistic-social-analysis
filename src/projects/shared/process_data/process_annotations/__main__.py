import os
from src.projects.shared.process_data.process_annotations.utils import convert_xml_to_coco_format
from src.projects.shared.process_data.process_annotations.create_database import create_db_annotations, create_db_table_video_name_id_mapping, correct_erronous_videos_in_db, add_annotations_to_db
from src.projects.shared.process_data.process_annotations.create_file_name_id_dict import create_file_name_id_dict
from src.projects.shared.process_data.process_annotations.convert_to_yolo import main as convert_to_yolo
from src.projects.shared.process_data.process_annotations.convert_to_mtcnn import main as convert_to_mtcnn
from src.projects.social_interactions.common.constants import DetectionPaths
from pathlib import Path

def main():
    os.environ['OMP_NUM_THREADS'] = '1'
    # Create a dictionary with the task name as key and a file id as valuey
    task_file_id_dict = create_file_name_id_dict(DetectionPaths.file_name_id_dict_path)
    # Create a database table for the video file names and ids
    create_db_table_video_name_id_mapping(task_file_id_dict)
    # Convert the XML annotations to COCO format and store the results in a database
    convert_xml_to_coco_format(DetectionPaths.annotations_xml_path, DetectionPaths.annotations_json_path)
    create_db_annotations()
    # Delete the erroneous videos in the database and add the new data form the indivual xml files
    correct_erronous_videos_in_db()
    
    # Add the correct annotations to the database
    for file_name in DetectionPaths.annotations_individual_folder_path.iterdir():
        add_annotations_to_db(file_name)
    
    # Convert the annotations to YOLO format and MTCNN format
    convert_to_yolo()
    convert_to_mtcnn()


if __name__ == "__main__":
    main()

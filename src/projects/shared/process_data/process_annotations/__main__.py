from src.projects.shared.process_data.process_annotations.utils import convert_xml_to_coco_format
from src.projects.shared.process_data.process_annotations.create_database import create_db_annotations, create_db_table_video_name_id_mapping
from src.projects.shared.process_data.process_annotations.create_file_name_id_dict import create_file_name_id_dict
from src.projects.shared.process_data.process_annotations.convert_to_yolo import main as convert_to_yolo
from src.projects.shared.process_data.process_annotations.convert_to_mtcnn import main as convert_to_mtcnn
from src.projects.social_interactions.common.constants import DetectionPaths


def main():
    # Create a dictionary with the task name as key and a dictionary as value
    task_file_id_dict = create_file_name_id_dict(DetectionPaths.file_name_id_dict_path)
    # Create a database table for the video file names and ids
    create_db_table_video_name_id_mapping(task_file_id_dict)
    # Convert the XML annotations to COCO format and store the results in a database
    convert_xml_to_coco_format(DetectionPaths.annotations_xml_path, DetectionPaths.annotations_json_path)
    create_db_annotations()
    # Convert the annotations to YOLO format and MTCNN format
    convert_to_yolo()
    convert_to_mtcnn()


if __name__ == "__main__":
    main()

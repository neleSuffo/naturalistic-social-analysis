from shared.process_annotations.utils import convert_xml_to_coco_format  
from projects.social_interactions.src.common.constants import DetectionPaths
from shared.process_annotations.create_database import create_database

if __name__ == "__main__":
    convert_xml_to_coco_format(DetectionPaths.annotations_xml_path, DetectionPaths.annotations_folder_path)
    create_database()
    
import subprocess
import yolo_person_face_gaze
from setup_detection_database import setup_detection_database

def main():
    # Setup the detection database which will hold the detection results
    setup_detection_database()
    # Run the detection pipeline
    yolo_person_face_gaze.main()

    
    

def main():
    # Setup the detection database which will hold the detection results
    setup_detection_database()
    # Run the detection pipeline
    yolo_person_face_gaze.main()
    
if __name__ == "__main__":
    main() 
    
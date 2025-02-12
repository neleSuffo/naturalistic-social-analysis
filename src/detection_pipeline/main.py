import detect_person_faces
from setup_detection_database import setup_detection_database

def main():
    
    # Setup the detection database which will hold the detection results
    setup_detection_database()
    detect_person_faces.main()
    
if __name__ == "__main__":
    main() 
    
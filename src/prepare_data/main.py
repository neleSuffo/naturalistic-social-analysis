import subprocess
import argparse
import logging
import os
from pathlib import Path
from typing import Optional
from prepare_data.prepare_training import main as prepare_training, balance_dataset
from prepare_data.process_videos import main as process_videos
from prepare_data.extract_faces import main as extract_faces
from prepare_data.extract_persons import main as extract_persons

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPipeline:
    """Handles the data preparation pipeline steps."""
    
    def __init__(self, model_target: Optional[str], yolo_target: Optional[str], threads: int = 2):
        self.model_target = model_target
        self.yolo_target = yolo_target
        os.environ['OMP_NUM_THREADS'] = str(threads)
        
    def process_videos(self):
        """Process videos into frames."""
        logging.info("Processing videos into frames...")
        process_videos()
        
    def extract_detections(self):
        """Extract person/face detections based on model target."""
        if self.model_target != "yolo" or self.yolo_target not in ["person", "face", "gaze"]:
            return
            
        logging.info(f"Extracting detections for {self.yolo_target}...")
        if self.yolo_target == "person":
            extract_persons()
        if self.yolo_target in ["face", "gaze"]:
            extract_faces()
            
    def process_annotations(self, setup_db: bool = False):
        """Process annotations for the specified target."""
        if not all([self.model_target, self.yolo_target]):
            raise ValueError("Model target and YOLO target required for annotation processing")
            
        logging.info(f"Processing annotations for {self.yolo_target}...")
        cmd = [
            'python', 
            '-m', 
            'prepare_data.process_annotations.__main__',
            self.model_target,
            self.yolo_target
        ]
        if setup_db:
            cmd.append('--setup_db')
            
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error processing annotations: {e}")
            raise
            
    def prepare_dataset(self):
        """Prepare and balance training dataset."""
        logging.info("Preparing training dataset...")
        prepare_training(self.model_target, self.yolo_target)
        balance_dataset(self.model_target, self.yolo_target)
        
    def run_pipeline(self, steps: list):
        """Run specified pipeline steps in order."""
        step_mapping = {
            'videos': self.process_videos,
            'detections': self.extract_detections,
            'annotations': lambda: self.process_annotations(setup_db=True),
            'dataset': self.prepare_dataset
        }
        
        for step in steps:
            if step in step_mapping:
                try:
                    step_mapping[step]()
                except Exception as e:
                    logging.error(f"Error in {step} step: {e}")
                    raise

def main():
    parser = argparse.ArgumentParser(description='Data preparation pipeline')
    parser.add_argument('--videos', action='store_true', help='Process videos into frames')
    parser.add_argument('--detections', action='store_true', help='Extract person/face detections')
    parser.add_argument('--annotations', action='store_true', help='Process annotations')
    parser.add_argument('--dataset', action='store_true', help='Prepare training dataset')
    parser.add_argument('--model_target', type=str, choices=['yolo', 'mtcnn', 'all'], help='Model target')
    parser.add_argument('--yolo_target', type=str, choices=['person_face', 'person_face_object', 'gaze', 'all', 'person', 'face'], help='YOLO target')
    parser.add_argument('--all', action='store_true', help='Run full pipeline')
    parser.add_argument('--threads', type=int, default=2, help='Number of threads')
    
    args = parser.parse_args()
    
    pipeline = DataPipeline(args.model_target, args.yolo_target, args.threads)
    
    if args.all:
        pipeline.run_pipeline(['videos', 'detections', 'annotations', 'dataset'])
    else:
        steps = []
        if args.videos:
            steps.append('videos')
        if args.detections:
            steps.append('detections')
        if args.annotations:
            steps.append('annotations')
        if args.dataset:
            steps.append('dataset')
        
        if steps:
            pipeline.run_pipeline(steps)
        else:
            parser.error("No pipeline steps specified")

if __name__ == "__main__":
    main()
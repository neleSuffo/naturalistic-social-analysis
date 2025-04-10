import subprocess
import argparse
import logging
import os
from pathlib import Path
from typing import Optional
from prepare_data.prepare_training import main as prepare_training
from prepare_data.process_videos import main as process_videos

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPipeline:
    """Handles the data preparation pipeline steps."""
    
    def __init__(self, yolo_target: Optional[str], threads: int = 2):
        self.yolo_target = yolo_target
        os.environ['OMP_NUM_THREADS'] = str(threads)
        
    def process_videos(self):
        """Process videos into frames."""
        logging.info("Processing videos into frames...")
        process_videos()
            
    def process_annotations(self, setup_db: bool = False):
        """Process annotations for the specified target."""
        if not self.yolo_target:
            raise ValueError("YOLO target required for annotation processing")
            
        logging.info(f"Processing annotations for {self.yolo_target}...")
        cmd = [
            'python', 
            '-m', 
            'prepare_data.process_annotations.__main__',
            '--yolo_target', self.yolo_target
        ]
        if setup_db:
            cmd.append('--setup_db')
            
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error processing annotations: {e}")
            raise
            
    def prepare_dataset(self):
        """Prepare training dataset."""
        logging.info("Preparing training dataset...")
        prepare_training(self.yolo_target)
        
    def run_pipeline(self, steps: list):
        """Run specified pipeline steps in order."""
        step_mapping = {
            'videos': self.process_videos,
            'annotations': lambda: self.process_annotations(setup_db=False),
            'annotations_db': lambda: self.process_annotations(setup_db=True),
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
    parser.add_argument('--annotations', action='store_true', help='Process annotations')
    parser.add_argument('--annotations_db', action='store_true', help='Setup annotations database')
    parser.add_argument('--dataset', action='store_true', help='Prepare training dataset')
    parser.add_argument('--yolo_target', type=str, choices=['person_face', 'person_face_object', 'gaze_cls', 'person_cls', 'face_cls'], help='YOLO target')
    parser.add_argument('--all', action='store_true', help='Run full pipeline')
    parser.add_argument('--threads', type=int, default=2, help='Number of threads')
    
    args = parser.parse_args()
    
    pipeline = DataPipeline(args.yolo_target, args.threads)
    
    if args.all:
        pipeline.run_pipeline(['videos', 'annotations', 'annotations_db', 'dataset'])
    else:
        steps = []
        if args.videos:
            steps.append('videos')
        if args.annotations:
            steps.append('annotations')
        if args.annotations_db:
            steps.append('annotations_db')
        if args.dataset:
            steps.append('dataset')
        
        if steps:
            pipeline.run_pipeline(steps)
        else:
            parser.error("No pipeline steps specified")

if __name__ == "__main__":
    main()
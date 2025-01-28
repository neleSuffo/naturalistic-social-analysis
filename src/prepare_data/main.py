import subprocess
import argparse
import logging
import os
from prepare_data.prepare_training import main as prepare_training
from prepare_data.process_videos import main as process_videos

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_process_annotations(model_target: str, yolo_target: str, setup_db=False):
    try:
        cmd = ['python', '-m', 'prepare_data.process_annotations.__main__', model_target, yolo_target]
        if setup_db:
            cmd.append('--setup_db')
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running process_annotations: {e}")

def main():
    parser = argparse.ArgumentParser(description='Data preparation pipeline')
    parser.add_argument('--videos', action='store_true', help='Run video processing')
    parser.add_argument('--annotations', action='store_true', help='Run annotation processing')
    parser.add_argument('--model_target', type=str, help='Model to convert to (e.g., "yolo", "mtcnn", "all")')
    parser.add_argument('--yolo_target', type=str, help='Target YOLO label (e.g., "face")')
    parser.add_argument('--training', action='store_true', help='Prepare training data')
    parser.add_argument('--all', action='store_true', help='Run all processes')
    parser.add_argument('--threads', type=int, default=2, help='Number of threads to use')
    
    args = parser.parse_args()
    os.num_threads = args.threads
    
    if args.all:
        run_process_videos()
        run_process_annotations(args.model_target, args.yolo_target)
        prepare_training()
    else:
        if args.videos:
            process_videos()
        if args.annotations:
            run_process_annotations()
        run_process_annotations(args.model_target, args.yolo_target)
        if args.training:
            prepare_training()
            
    logging.info("Data preparation complete.")
    
if __name__ == "__main__":
    main()

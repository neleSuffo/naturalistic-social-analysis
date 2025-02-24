import subprocess
import argparse
import logging
import os
from prepare_data.prepare_training import main as prepare_training, balance_dataset
from prepare_data.process_videos import main as process_videos
from prepare_data.extract_faces import main as extract_faces

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_process_annotations(model_target: str, yolo_target: str, setup_db=False):
    """
    Run the process_annotations script as a subprocess.
    
    Parameters
    ----------
    model_target : str
        Model to convert to (e.g., "yolo", "mtcnn", "all")
    yolo_target : str
        Target YOLO label (e.g., "person_face", "person_face_object", "gaze")
    setup_db : bool
        Whether to set up the database, defaults to False
    """
    try:
        cmd = ['python', '-m', 'prepare_data.process_annotations.__main__', model_target, yolo_target]
        if setup_db:
            cmd.append('--setup_db')
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error while running process_annotations: {e}")
        raise

def main():
    
    os.environ['OMP_NUM_THREADS'] = str(args.threads)

    if args.all:
        process_videos()
        if args.model_target == "yolo" and args.model_target == "gaze":
            extract_faces()
        run_process_annotations(args.model_target, args.yolo_target, setup_db=True)
        prepare_training(args.model_target, args.yolo_target)
        balance_dataset(args.model_target, args.yolo_target)

    else:
        if args.videos:
            # Extract rawframes from videos
            process_videos()
        if args.annotations:
            if not args.model_target or not args.yolo_target:
                parser.error("--annotations requires --model_target and --yolo_target.")
            run_process_annotations(args.model_target, args.yolo_target, setup_db=args.setup_db)
        if args.face_rawframes:
            # Extract faces from rawframes
            extract_faces()
        if args.training:
            prepare_training(args.model_target, args.yolo_target)
        if args.balanced:
            balance_dataset(args.model_target, args.yolo_target)

    logging.info("Data preparation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data preparation pipeline')
    parser.add_argument('--videos', action='store_true', help='Run video processing')
    parser.add_argument('--face_rawframes', action='store_true', help='Run face rawframes extraction')
    parser.add_argument('--annotations', action='store_true', help='Run annotation processing')
    parser.add_argument('--model_target', type=str, help='Model to convert to (e.g., "yolo", "mtcnn", "all")')
    parser.add_argument('--yolo_target', type=str, help='Target YOLO label ("person_face", "person_face_object", "gaze")')
    parser.add_argument('--setup_db', action='store_true', default=False, help='Whether to set up the database (default: False)')
    parser.add_argument('--training', action='store_true', help='Prepare training data')
    parser.add_argument('--balanced', action='store_true', help='Balance the dataset into equal number of frames with and without class')
    parser.add_argument('--all', action='store_true', help='Run all processes')
    parser.add_argument('--threads', type=int, default=2, help='Number of threads to use')

    args = parser.parse_args()
    main()
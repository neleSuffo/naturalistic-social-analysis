import os
import json
import logging
import torch
import torch.nn as nn
import utils
from ultralytics import YOLO
from multiprocessing.pool import ThreadPool
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN
from pathlib import Path
from timeit import default_timer as timer
from models.mtcnn.run_mtcnn import run_mtcnn
from models.yolo_inference.run_yolo import run_yolo
#fromvmodels.resnet.train_gaze_model import GazeEstimationModel
from models.vtc import detect_voices
from constants import (
    DetectionPaths,
    ResNetPaths,
    YoloPaths,
)
from config import (
    DetectionParameters,
)
from multiprocessing import Value
from typing import Dict
from threading import Lock
 
 # Set up logging
logging.basicConfig(level=logging.INFO)   

class SharedResources:
    def __init__(self):
        #  Initialize a lock for thread-safe operations
        self.lock = Lock()
        # Initialize shared annotation_id and image_id
        # This variable is used to assign unique IDs to the annotations when running the detection models in parallel
        self.annotation_id = Value("i", 1)
        self.image_id = Value("i", 1)
        self.video_file_ids_dict = {}  
        
    def create_video_id_mapping(self):
        return utils.create_video_to_id_mapping(
            [video_file.stem for video_file in DetectionPaths.videos_input_dir.iterdir()
             if video_file.suffix.lower() == DetectionParameters.video_file_extension]
        )


class OutputMerger:
    def __init__(self):
        self.combined_output = {
            DetectionParameters.output_key_videos: [],
            DetectionParameters.output_key_images: [],
            DetectionParameters.output_key_categories: [],
        }

    def merge(self, detection_output: dict):
        # Get existing video IDs to avoid duplicates
        existing_video_ids = {video["id"] for video in self.combined_output[DetectionParameters.output_key_videos]}
        
        # Add new videos to the combined COCO output
        for video in detection_output.get(DetectionParameters.output_key_videos, []):
            if video["id"] not in existing_video_ids:
                self.combined_output[DetectionParameters.output_key_videos].append(video)
        
        # Add annotations and images
        self.combined_output[DetectionParameters.output_key_images].extend(detection_output.get(DetectionParameters.output_key_images, []))
        self.combined_output[DetectionParameters.output_key_categories].extend(detection_output.get(DetectionParameters.output_key_categories, []))
    
    def get_combined_output(self):
        return self.combined_output
    

class Detector:
    def __init__(self, shared_resources: SharedResources, output_merger: OutputMerger):
        self.shared_resources = shared_resources
        self.output_merger = output_merger
        
        # Initialize processors
        self.yolo_processor = YOLOProcessor(output_merger, shared_resources)
        self.mtcnn_processor = MTCNNProcessor(output_merger, shared_resources)
        self.voice_processor = VoiceTypeProcessor(output_merger, shared_resources)
        
    def process_video_file(self, video_file: Path, detections: Dict[str, bool]) -> None:
        """
        This function processes a video file by performing the specified detections.
        
        Parameters
        ----------
        video_file : Path
            the path to the video file to process
        detections : dict
            the detections to perform
        """
        # Get the video file name
        file_name_short = video_file.stem
        logging.info(f"Processing {file_name_short}...")

        # Add the video file to the output
        self.output_merger.combined_output[DetectionParameters.output_key_videos].append(
            {
                "id": self.shared_resources.video_file_ids_dict[file_name_short], 
                "file_name": video_file.name,
            }
        )

        # Run the desired detections
        if detections.get("person", False):
            self.yolo_processor.run_person_detection(video_file)
        if detections.get("face", False):
            self.mtcnn_processor.run_face_detection(video_file)
        if detections.get("voice", False):
            self.voice_processor.process_all()
            
    def wrapper(self, args: tuple) -> None:
        """
        This function is a wrapper for the process_video_file function.

        Parameters
        ----------
        args : tuple
            the arguments for the process_video_file function

        """
        video_file, detections = args
        return self.process_video_file(video_file, detections)

    def run_detection(self, detections: dict, batch_size) -> dict:
        """
        This function runs the detection models in batches.
        
        Parameters
        ----------
        detections : dict
            the detections to perform
        batch_size : int
            the batch size for processing videos
        """
        # Get a list of all video files in the folder
        video_files = [
            video_f 
            for video_f in DetectionPaths.videos_input_dir.iterdir() 
            if video_f.suffix.lower() == DetectionParameters.video_file_extension
        ]
        # Process the videos in batches
        try:
            total_batches = (len(video_files) + batch_size - 1) // batch_size
            for i in range(0, len(video_files), batch_size):
                batch = video_files[i:i + batch_size]
                batch_number = (i // batch_size) + 1
                logging.info(f"Processing batch {batch_number}/{total_batches} with {len(batch)} videos...")
                # Process the videos in parallel
                with ThreadPool() as pool:
                    pool.map(
                        self.wrapper,
                        [
                            (video_file, detections) 
                            for video_file in batch
                        ],
                    )
        except Exception as e:
            logging.error(f"An error occurred during detection: {e}")
            raise
        
        return self.output_merger.get_combined_output()           
           
class MTCNNProcessor:
    def __init__(self, output_merger: OutputMerger, shared_resources):
        self.output_merger = output_merger
        # Use the shared resources
        self.annotation_id = shared_resources.annotation_id
        self.image_id = shared_resources.image_id
        # Pass the MTCNN model instance
        # keep_all=True to get all faces in the image
        self.model = MTCNN(keep_all=True, device='cuda:0')
        
        # Load the gaze estimation model
        # self.gaze_model = GazeEstimationModel(pretrained=False)
        # if ResNetParameters.trained_model_path.exists():
        #     self.gaze_model.load_state_dict(torch.load(ResNetParameters.trained_model_path))
        # # Set the model to evaluation mode
        # self.gaze_model.eval()
        # # Move the model to the GPU
        # self.gaze_model = self.gaze_model.to('cuda:0')
        # # Define the image transformation
        # self.transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])

    def run_face_detection(self, video_file):
        logging.info("Running face detection...")
        detection_output = run_mtcnn(
            video_file, 
            self.annotation_id, 
            self.image_id, 
            self.model,
        )
        
        if detection_output:
            # For each detected face, estimate gaze
            for annotation in detection_output[DetectionParameters.output_key_annotations]:
                # Get the bounding box and frame ID
                bbox = annotation["bbox"]
                frame_id = annotation["frame_id"]
                
                # Find the corresponding image file
                image_entry = next((image for image in detection_output[DetectionParameters.output_key_images] if image["id"] == frame_id), None)
                if image_entry is not None:
                    image_file_name = image_entry["file_name"]
                    image_file_path = DetectionPaths.images_input_dir / image_file_name
                    try:
                        # Load the image and crop the face
                        frame = Image.open(image_file_path)
                        face = frame.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
                        face = self.transform(face).unsqueeze(0).to('cuda:0')

                        # Predict gaze direction
                        with torch.no_grad():
                            gaze_prob = self.gaze_model(face).item()
                            annotation['gaze'] = 'looking' if gaze_prob > 0.5 else 'not looking'
                        
                    except FileNotFoundError:
                        logging.error(f"Image file not found: {image_file_name}")
                    except Exception as e:
                        logging.error(f"Error processing image {image_file_name}: {e}")
                else:
                    print(f"Image entry not found for frame ID {frame_id}")
            
            # Merge the results into the combined output
            self.output_merger.merge(detection_output)


class YOLOProcessor:
    def __init__(self, output_merger: OutputMerger, shared_resources):
        self.output_merger = output_merger
        # Use the shared annotation_id and image_id
        self.annotation_id = shared_resources.annotation_id
        self.image_id = shared_resources.image_id
        self.model = YOLO(YoloPaths.trained_weights_path)
        # Retrieve the YOLO model class names and their IDs
        self.class_dict = self.model.names  # dictionary {id: name}    
        # Logging to confirm model loading
        logging.info("YOLO model loaded successfully from %s", YoloPaths.trained_weights_path)
        
    def run_person_detection(self, video_file):
        logging.info("Running person detection...")
        detection_output = run_yolo(
            video_file, 
            self.model,
        )
        #Is None when generate_output_video =True
        if detection_output:
            # Add each class to the combined output categories
            for class_id, class_name in self.class_dict.items():
                detection_output[DetectionParameters.output_key_categories].append({
                    "id": class_id,
                    "name": class_name
                })
                # Merge the results into the combined output
                self.output_merger.merge(detection_output)
    
class VoiceTypeProcessor:
    def __init__(self, output_merger: OutputMerger, shared_resources):
        self.output_merger = output_merger
        # Use the shared annotation_id and image_id
        self.annotation_id = shared_resources.annotation_id
        self.image_id = shared_resources.image_id
    
    def extract_audio_from_videos(self):
        """This function extracts audio from videos for voice detection."""
        logging.info("Extracting audio for voice detection...")
        utils.extract_audio_from_videos_in_folder(DetectionPaths.videos_input)
        
    def run_voice_detection(self):
        """This function runs voice detection on the extracted audio files.

        Returns
        -------
        dict
            the detection results in COCO format
        """
        logging.info("Running voice detection...")
        detection_output = detect_voices.run_voice_detection(self.video_file_ids_dict, self.annotation_id, self.image_id)

        # Merge the results into the combined output
        self.output_merger.merge(detection_output)
        
    def process_all(self):
        """This function extracts audio from videos and runs voice detection."""
        self.extract_audio_from_videos()
        detection_output = self.run_voice_detection()
        return detection_output 


def main(detections: dict, batch_size: int) -> None:
    """
    This function runs the social interactions detection pipeline.

    Parameters
    ----------
    detections : dict
        the detections to perform
    batch_size : int
        the batch size for processing videos
    """
    logging.info("Starting detection process...")
    start_time = timer()
    output_merger = OutputMerger()
    shared_resources = SharedResources()
        
    # Create video file IDs if they don't already exist
    if not shared_resources.video_file_ids_dict:
        video_file_mapping = shared_resources.create_video_id_mapping()
        logging.info(f"Video file IDs mapping created")
        shared_resources.video_file_ids_dict = video_file_mapping

    # Instantiate the detector and run the detection process
    detector = Detector(shared_resources, output_merger)
    combined_output = detector.run_detection(detections, batch_size)
    
    # Write the combined output to a JSON file
    with DetectionPaths.combined_json_output_path.open('w') as file:
        json.dump(combined_output, file, indent=4)    
    
    end_time = timer()
    runtime = end_time - start_time
    logging.info(f"Detection process completed. Runtime: {runtime} seconds")


if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    # Define which detections to perform
    detections = {"person": True, "face": False, "voice": False}
    batch_size = 2
    main(detections, batch_size)

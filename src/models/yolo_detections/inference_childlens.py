import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
from estimate_proximity import get_proximity
import logging

def analyze_videos_for_faces(video_folder: Path, 
                           detection_model: YOLO,
                           num_videos: int = 10,
                           frame_skip: int = 10):
    """Analyze first 5 videos and store face detections with proximity values."""
    face_detections = {
        'adult face': [],  # for class 3.0
        'infant/child face': []   # for class 2.0
    }
    
    videos = list(video_folder.glob("*.MP4"))[:num_videos]
    
    for video_path in videos:
        logging.info(f"Processing {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))
        frame_number = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_number % frame_skip == 0:
                results = detection_model(frame)
                boxes = results[0].boxes
                
                for box in boxes:
                    if box.cls in [2.0, 3.0]:  # face classes
                        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                        face_type = 'adult face' if box.cls == 3.0 else 'infant/child face'
                        
                        # Calculate proximity
                        bbox = [x_min, y_min, x_max, y_max]
                        proximity = get_proximity(bbox, f"{face_type} face")
                        
                        # Store detection info
                        face_detections[face_type].append({
                            'video_name': video_path.stem,
                            'frame_number': frame_number,
                            'proximity': proximity
                        })
            
            frame_number += 1
        
        cap.release()
    
    return face_detections

def sample_faces_by_proximity(face_detections, bins=10, samples_per_bin=10):
    """Sample faces evenly across proximity bins."""
    sampled_faces = {
        'adult face': [],
        'infant/child face': []
    }
    
    bin_edges = np.linspace(0, 1, bins+1)
    
    for face_type in ['adult face', 'infant/child face']:
        df = pd.DataFrame(face_detections[face_type])
        
        for i in range(len(bin_edges)-1):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i+1]
            
            # Get faces in this bin
            bin_faces = df[
                (df['proximity'] >= bin_start) & 
                (df['proximity'] < bin_end)
            ]
            
            if len(bin_faces) > 0:
                # Sample faces from this bin
                sampled = bin_faces.sample(
                    n=min(samples_per_bin, len(bin_faces)),
                    replace=False
                )
                
                for _, row in sampled.iterrows():
                    image_path = f"{row['video_name']}_{row['frame_number']:06d}"
                    sampled_faces[face_type].append({
                        'image_path': image_path,
                        'proximity': row['proximity'],
                        'proximity_bin': f"{bin_start:.1f}-{bin_end:.1f}"
                    })
    
    return sampled_faces

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Initialize model
    model = YOLO('/home/nele_pauline_suffo/models/yolo11_all_detection.pt')
    
    # Analyze videos
    video_folder = Path('/home/nele_pauline_suffo/ProcessedData/childlens_videos')
    face_detections = analyze_videos_for_faces(video_folder, model)
    
    # Convert to DataFrames and save
    output_dir = Path('/home/nele_pauline_suffo/outputs/proximity_sampled_frames')
    output_dir.mkdir(exist_ok=True)
    
    for face_type in ['adult face', 'infant/child face']:
        df = pd.DataFrame(face_detections[face_type])
        output_file = output_dir / f"{face_type}_detections.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} {face_type} detections to {output_file}")

if __name__ == "__main__":
    main()
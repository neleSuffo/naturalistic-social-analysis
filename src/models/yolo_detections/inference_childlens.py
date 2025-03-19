import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
from estimate_proximity import get_proximity
import logging

def get_safe_filename(face_type: str) -> str:
    """Convert face type to a safe filename by replacing slashes with underscores."""
    return face_type.replace('/', '_')

def analyze_videos_for_faces(video_folder: Path, 
                           detection_model: YOLO,
                           num_videos: int = 20,
                           frame_skip: int = 30):
    """Analyze first 5 videos and store face detections with proximity values."""
    face_detections = {
        'adult face': [],  # for class 3.0
        'infant/child face': []   # for class 2.0
    }
    
    videos = list(video_folder.glob("*.MP4"))[:num_videos]
    all_videos = list(video_folder.glob("*.MP4"))
    videos = all_videos[15:15 + num_videos]

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
                        proximity = get_proximity(bbox, face_type)
                        
                        # Store detection info
                        face_detections[face_type].append({
                            'video_name': video_path.stem,
                            'frame_number': frame_number,
                            'proximity': proximity
                        })
            
            frame_number += 1
        
        cap.release()
    
    return face_detections

def sample_faces_by_proximity(df: pd.DataFrame, bins=10, samples_per_bin=10):
    """Sample faces evenly across proximity bins."""
    # Create bin labels
    bin_edges = np.linspace(0, 1, bins+1)
    df['proximity_bin'] = pd.cut(df['proximity'], 
                                bins=bin_edges, 
                                labels=[f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" 
                                       for i in range(len(bin_edges)-1)])
    
    # Sample from each bin
    sampled_dfs = []
    for bin_label in df['proximity_bin'].unique():
        bin_data = df[df['proximity_bin'] == bin_label]
        if len(bin_data) > 0:
            sampled = bin_data.sample(n=min(samples_per_bin, len(bin_data)), 
                                    replace=False)
            sampled_dfs.append(sampled)
    
    # Combine all samples and keep original columns
    if sampled_dfs:
        final_df = pd.concat(sampled_dfs)[['video_name', 'frame_number', 'proximity']]
        return final_df
    return pd.DataFrame(columns=['video_name', 'frame_number', 'proximity'])

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Initialize model
    model = YOLO('/home/nele_pauline_suffo/models/yolo11_all_detection.pt')
    
    # Analyze videos
    video_folder = Path('/home/nele_pauline_suffo/ProcessedData/childlens_videos')
    face_detections = analyze_videos_for_faces(video_folder, model)
    
    # Create output directory
    output_dir = Path('/home/nele_pauline_suffo/outputs/proximity_sampled_frames')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrames and save
    for face_type in ['adult face', 'infant/child face']:
        df = pd.DataFrame(face_detections[face_type])
        safe_name = get_safe_filename(face_type)
        output_file = output_dir / f"{safe_name}_detections.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} {face_type} detections to {output_file}")

def main_two():
    logging.basicConfig(level=logging.INFO)
    
    data_dir = Path('/home/nele_pauline_suffo/outputs/proximity_sampled_frames')
    
    # Process each face type
    for face_type in ['adult_face', 'infant_child_face']:
        # Load detections
        input_file = data_dir / f"{face_type}_detections.csv"
        df = pd.read_csv(input_file)
        print(f"\nLoaded {len(df)} {face_type} detections")
        
        # Sample faces
        sampled_df = sample_faces_by_proximity(df)
        
        if sampled_df.empty:
            print(f"No {face_type}s were sampled!")
            continue
        
        # Count samples per bin for display
        bin_edges = np.linspace(0, 1, 11)
        bin_counts = pd.cut(sampled_df['proximity'], 
                           bins=bin_edges, 
                           labels=[f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" 
                                  for i in range(len(bin_edges)-1)]).value_counts()
        print("\nSamples per proximity bin:")
        print(bin_counts)
        
        # Save sampled faces
        output_file = data_dir / f"{face_type}_samples.csv"
        sampled_df.to_csv(output_file, index=False)
        print(f"Saved {len(sampled_df)} samples to {output_file}")
        
if __name__ == "__main__":
    main()
    #main_two()
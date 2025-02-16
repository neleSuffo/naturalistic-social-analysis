import os
import re
import shutil
from collections import defaultdict
import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN

def get_images_by_id(image_directory, pattern):
    images_by_id = defaultdict(list)
    for filename in os.listdir(image_directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            match = pattern.search(filename)
            if match:
                unique_id = match.group()
                image_path = os.path.join(image_directory, filename)
                images_by_id[unique_id].append(image_path)
    return images_by_id

def extract_embeddings(image_paths):
    embeddings = []
    valid_image_paths = []
    for image_path in image_paths:
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            embeddings.append(face_encodings[0])
            valid_image_paths.append(image_path)
    return np.array(embeddings), valid_image_paths

def cluster_faces(embeddings, eps=0.5, min_samples=5):
    dbscan = DBSCAN(metric='euclidean', eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(embeddings)
    return labels

def process_clusters(images_by_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for unique_id, image_paths in images_by_id.items():
        print(f'Processing {unique_id} with {len(image_paths)} images.')
        
        # Extract embeddings
        embeddings, valid_image_paths = extract_embeddings(image_paths)
        if len(embeddings) == 0:
            print(f'No valid faces found for {unique_id}. Skipping.')
            continue
        
        # Cluster faces
        labels = cluster_faces(embeddings)
        
        # Create a directory for the current unique ID
        id_dir = os.path.join(output_dir, unique_id)
        os.makedirs(id_dir, exist_ok=True)
        
        # Create subdirectories for each cluster and copy images
        for label in set(labels):
            cluster_dir = os.path.join(id_dir, f'cluster_{label}')
            os.makedirs(cluster_dir, exist_ok=True)
            for image_path, cluster_label in zip(valid_image_paths, labels):
                if cluster_label == label:
                    shutil.copy(image_path, cluster_dir)
        
        print(f'Clustering for {unique_id} completed. {len(set(labels))} clusters found.')

def main():
    image_directory = '/home/nele_pauline_suffo/ProcessedData/quantex_gaze_input'
    output_dir = '/home/nele_pauline_suffo/outputs/clustered_faces'
    pattern = re.compile(r'quantex_at_home_id\d+')
    
    images_by_id = get_images_by_id(image_directory, pattern)
    process_clusters(images_by_id, output_dir)

if __name__ == '__main__':
    main()
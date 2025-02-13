import shutil
import random
import logging
import os
from collections import defaultdict
from pathlib import Path
from constants import DetectionPaths, YoloPaths
from config import TrainingConfig
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_class_distribution(label_path):
    """Get the ratio of images with persons and faces to total images."""
    total_images = [f.name for f in label_path.glob('*.txt')]
    images_with_person = set()
    images_with_face = set()
    images_with_child = set()
    
    for label_file in total_images:
        with open(os.path.join(label_path, label_file), 'r') as f:
            labels = f.readlines()
            # Assuming person is class 0 and face is class 1 in YOLO format
            for label in labels:
                class_id = int(label.split()[0])
                if class_id == 0:  # Person class
                    images_with_person.add(label_file)
                elif class_id == 1:  # Face class
                    images_with_face.add(label_file)
                elif class_id == 2: # child_body_parts
                    images_with_child.add(label_file)
    
    total_images_count = len(total_images)
    person_ratio = len(images_with_person) / total_images_count
    face_ratio = len(images_with_face) / total_images_count
    child_ratio = len(images_with_child) / total_images_count
    
    logging.info(f"Images with at least one persons: {person_ratio:.2f}, Images with at least one faces: {face_ratio:.2f}, Images with at child body parts: {child_ratio:.2f}")
    return person_ratio, face_ratio, child_ratio, total_images_count

def stratified_pf_split(total_images, label_path, train_ratio=TrainingConfig.train_test_split_ratio):
    """Stratified split of images based on presence of persons and faces, ensuring the same ratio across splits."""
    validation_ratio = (1 - train_ratio) / 2
    test_ratio = validation_ratio
    person_ratio, face_ratio, _, total_images_count = get_class_distribution(label_path)
    
    # Calculate the number of images for each split
    total_train = int(train_ratio * total_images_count)
    total_val = int(validation_ratio * total_images_count)
    total_test = total_images_count - total_train - total_val
    
    # Separate images based on the presence of persons and faces
    images_with_person = [f for f in total_images if is_image_with_person(f, label_path)]
    images_with_face = [f for f in total_images if is_image_with_face(f, label_path)]
    images_without_person_or_face = [f for f in total_images if not is_image_with_person(f, label_path) and not is_image_with_face(f, label_path)]
    
    # Sample images to match the desired ratio for each class
    train_images = images_with_person[:int(total_train * person_ratio)] + images_with_face[:int(total_train * face_ratio)]
    validation_images = images_with_person[int(total_train * person_ratio):int((total_train + total_val) * person_ratio)] + \
                        images_with_face[int(total_train * face_ratio):int((total_train + total_val) * face_ratio)]
    test_images = images_with_person[int((total_train + total_val) * person_ratio):] + \
                  images_with_face[int((total_train + total_val) * face_ratio):]

    # Move images into their respective folders
    move_images(train_images, "train", label_path)
    move_images(validation_images, "val", label_path)
    move_images(test_images, "test", label_path)

def is_image_with_person(image_file, label_path):
    """Check if the image contains a person by checking its label file."""
    label_file = os.path.splitext(image_file)[0] + ".txt"
    with open(os.path.join(label_path, label_file), 'r') as f:
        labels = f.readlines()
        for label in labels:
            if int(label.split()[0]) == 0:
                return True
    return False

def is_image_with_face(image_file, label_path):
    """Check if the image contains a face by checking its label file."""
    label_file = os.path.splitext(image_file)[0] + ".txt"
    with open(os.path.join(label_path, label_file), 'r') as f:
        labels = f.readlines()
        for label in labels:
            if int(label.split()[0]) == 1: 
                return True
    return False

def move_images(images, split_type, label_path):
    """Move image and corresponding label files to the correct split directory."""
    for image in images:
        image_src = os.path.join(YoloPaths.person_images_input_dir, image)
        label_src = os.path.join(label_path, os.path.splitext(image)[0] + ".txt")
        image_dst = os.path.join(YoloPaths.person_face_output_dir, split_type, "images", image)
        label_dst = os.path.join(YoloPaths.person_face_output_dir, split_type, "labels", os.path.splitext(image)[0] + ".txt")

        shutil.copy(image_src, image_dst)
        shutil.copy(label_src, label_dst)

def split_yolo_data(total_images, label_path, yolo_target):
    """Handle the YOLO data split based on the target model type."""
    if yolo_target == "person_face":
        # Perform stratified split for person and face categories
        stratified_pf_split(total_images, label_path)
    elif yolo_target == "gaze":
        # Perform stratified split for gaze category
        stratified_gaze_split(total_images, label_path)

def main(model_target, yolo_target):
    if model_target == "yolo":
        label_path = YoloPaths.person_labels_input_dir if yolo_target == "person_face" else YoloPaths.gaze_labels_input_dir
        image_path = YoloPaths.person_images_input_dir if yolo_target == "person_face" else YoloPaths.gaze_images_input_dir

        total_images = [f.name for f in Path(image_path).glob('*.jpg')]
       
        # Perform the split and copy images/labels to the correct directories
        split_yolo_data(total_images, label_path, yolo_target)
        logging.info("Dataset preparation for YOLO completed.")
    
    elif model_target == "other_model":
        pass
    else:
        logging.error("Unsupported model target specified!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for model training.")
    parser.add_argument("--model_target", choices=["yolo", "other_model"], required=True, help="Specify the model type")
    parser.add_argument("--yolo_target", choices=["person+face", "gaze"], required=True, help="Specify the YOLO target type")

    args = parser.parse_args()
    main(args.model_target, args.yolo_target)
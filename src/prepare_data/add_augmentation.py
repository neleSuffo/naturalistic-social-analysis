import cv2
import numpy as np
import random
import shutil
import logging
from pathlib import Path
import albumentations as A

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def oversample_with_augmentation(train_dir: Path, target_ratio=0.5):
    # Create new augmented train directory
    augmented_train_dir = train_dir.parent / 'train_augmented'
    augmented_train_dir.mkdir(exist_ok=True)
    
    # Copy original class directories
    augmented_gaze_dir = augmented_train_dir / "gaze"
    augmented_no_gaze_dir = augmented_train_dir / "no_gaze"
    
    # Copy existing files
    if augmented_gaze_dir.exists():
        shutil.rmtree(augmented_gaze_dir)
    if augmented_no_gaze_dir.exists():
        shutil.rmtree(augmented_no_gaze_dir)
        
    shutil.copytree(train_dir / "gaze", augmented_gaze_dir)
    shutil.copytree(train_dir / "no_gaze", augmented_no_gaze_dir)
    
    # Get original counts
    gaze_images = list(augmented_gaze_dir.glob('*'))
    no_gaze_images = list(augmented_no_gaze_dir.glob('*'))
    
    gaze_count = len(gaze_images)
    no_gaze_count = len(no_gaze_images)
    
    logging.info(f"Original counts - Gaze: {gaze_count}, No Gaze: {no_gaze_count}")
    
    target_gaze_count = int((no_gaze_count * target_ratio) / (1 - target_ratio))
    images_to_add = target_gaze_count - gaze_count
    
    logging.info(f"Adding {images_to_add} augmented gaze images")
    
    aug_pipeline = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.PiecewiseAffine(p=0.3),
        ], p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
        ], p=0.3),
    ])
    
    for i in range(images_to_add):
        src_image = random.choice(gaze_images)
        
        # Read image
        image = cv2.imread(str(src_image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation
        augmented = aug_pipeline(image=image)
        augmented_image = augmented['image']
        
        # Save augmented image
        dst_image = augmented_gaze_dir / f"augmented_gaze_{i}_{src_image.name}"
        cv2.imwrite(str(dst_image), cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
    
    # Recount images
    gaze_count = len(list(augmented_gaze_dir.glob('*')))
    no_gaze_count = len(list(augmented_no_gaze_dir.glob('*')))
    
    logging.info(f"New counts - Gaze: {gaze_count}, No Gaze: {no_gaze_count}")
    logging.info(f"New ratio: {gaze_count / (gaze_count + no_gaze_count):.2f}")

# Usage
train_dir = Path('/home/nele_pauline_suffo/ProcessedData/yolo_gaze_input/train')
oversample_with_augmentation(train_dir, target_ratio=0.5)
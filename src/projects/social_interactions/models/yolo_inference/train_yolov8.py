import os
from src.projects.social_interactions.common.constants import YoloPaths as YP
from src.projects.social_interactions.config import YoloConfig as YC
from ultralytics import YOLO

def main():
    # Load a pre-trained YOLOv8 model
    model = YOLO('yolov8x.pt') 

    results = model.train(
        data=str(YP.data_config_path), 
        epochs=YC.num_epochs, 
        batch=YC.batch_size, 
        imgsz=YC.img_size, 
        device=(0, 1), # Train on GPU 0 and 1
        project="outputs/yolov8/train",         
        name="exp",
        classes=[0],  # Focus on training the 'person' and 'reflection' class
        lr0=0.0005,  # Initial learning rate
        lrf=0.005,   # Final learning rate as a fraction of the initial learning rate (e.g., 0.01 times lr0)
        cos_lr=True,  #Apply cosine learning rate decay
        warmup_epochs=5,  # Add warm-up period
        augment=True,  # Enable data augmentation
        patience=15,  # Set patience for early stopping
        plots=True,  # Enable plots
    )

    print("Training results:", results)

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '10'
    main()
    
    
#run10
"""
run13: mAP@0.5=0.803
results = model.train(
    data=str(Yolo.data_config), 
    epochs=Yolo.epochs, 
    batch=Yolo.batch_size, 
    imgsz=Yolo.img_size, 
    device=(0, 1), # Train on GPU 0 and 1
    project="outputs/yolov8/train",         
    name="exp",
    classes=[0],  # Focus on training the 'person' and 'reflection' class
    lr0=0.0005,  # Initial learning rate
    lrf=0.005,   # Final learning rate as a fraction of the initial learning rate (e.g., 0.01 times lr0)
    cos_lr=True,  #Apply cosine learning rate decay
    warmup_epochs=5,  # Add warm-up period
    augment=True,  # Enable data augmentation
    patience=15,  # Set patience for early stopping
    plots=True,  # Enable plots
)


run 10
lr0=0.0005,  # Initial learning rate
lrf=0.005,   # Final learning rate as a fraction of the initial learning rate (e.g., 0.01 times lr0)
patience=20,  # Set patience for early stopping
"""


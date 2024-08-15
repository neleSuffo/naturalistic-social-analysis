import os
from src.projects.social_interactions.common.constants import YoloParameters as Yolo
from ultralytics import YOLO

def main():
    # Load a pre-trained YOLOv8 model
    model = YOLO('yolov8l.pt') 

    results = model.train(
        data=str(Yolo.data_config), 
        epochs=Yolo.epochs, 
        batch=Yolo.batch_size, 
        imgsz=Yolo.img_size, 
        device=(0, 1), # Train on GPU 0 and 1
        project="outputs/yolov8/train",         
        name="exp",
        classes=[0,1],  # Focus on training the 'person' and 'reflection' class
        lr0=0.001,  # Initial learning rate
        lrf=0.01,   # Final learning rate as a fraction of the initial learning rate (e.g., 0.01 times lr0)
        cos_lr=True,  #Apply cosine learning rate decay
        augment=True,  # Enable data augmentation
        patience=10,  # Set patience for early stopping
        plots=True,  # Enable plotsy
    )

    print("Training results:", results)

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '10'
    main()
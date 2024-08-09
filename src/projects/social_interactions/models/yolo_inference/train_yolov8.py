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
        device=[0, 1], # Train on GPU 0 and 1
        project="outputs/yolov8/train",         
        name="exp"
    )

    print("Training results:", results)

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '10'
    main()
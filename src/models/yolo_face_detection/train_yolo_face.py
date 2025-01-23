from ultralytics import YOLO
from constants import YoloPaths

# Load the YOLO model
model = YOLO("/home/nele_pauline_suffo/models/yolov8_github_face.pt")  # Use pretrained YOLOv8 model

# Train the model with a cosine annealing learning rate scheduler
model.train(
    data=str(YoloPaths.face_data_config_path),
    epochs=50,  # Total number of epochs
    imgsz=640,  # Image size
    batch=16,   # Batch size
    project=str(YoloPaths.face_output_dir),  # Output directory
    name="yolo_face_finetune",  # Experiment name
    lr0=0.01,  # Initial learning rate
    lrf=0.001,  # Final learning rate after scheduling
    cos_lr=True,  # Use cosine annealing for learning rate scheduling,
    device=0,  # GPU (use "cpu" for CPU training)
)
from ultralytics import YOLO
from constants import YoloPaths

# Load the YOLO model
model = YOLO(str(YoloPaths.face_trained_weights_path))  # Use pretrained YOLOv8 model

# Train the model on your dataset
model.train(
    data=str(YoloPaths.face_data_config_path),  # Dataset configuration file
    epochs=50,  # Adjust epochs as needed
    imgsz=640,  # Image size
    batch=16,   # Batch size
    project=str(YoloPaths.face_output_dir),  # Output directory
    name="yolo_face_finetune", # Experiment name
    device=0   # GPU (use "cpu" for CPU training)
)
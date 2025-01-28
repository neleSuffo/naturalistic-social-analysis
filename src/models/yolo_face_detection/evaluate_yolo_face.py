from ultralytics import YOLO
from constants import YoloPaths
from datetime import datetime
# Load a model
model = YOLO(YoloPaths.face_trained_weights_path)

folder_name = "face_validation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
# Validate the model
metrics = model.val(data=str(YoloPaths.face_data_config_path),
                    save_json=True, 
                    iou=0.5, 
                    plots=True, 
                    project=str(YoloPaths.face_output_dir),
                    name=folder_name)
# Extract precision and recall
precision = metrics.results_dict['metrics/precision(B)']
recall = metrics.results_dict['metrics/recall(B)']

# save precision and recall to a file
with open(YoloPaths.face_output_dir / folder_name / "precision_recall.txt", "w") as f:
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
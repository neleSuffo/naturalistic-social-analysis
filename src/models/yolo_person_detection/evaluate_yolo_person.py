import logging
from ultralytics import YOLO
from constants import YoloPaths
from datetime import datetime
# Load a model
#model = YOLO(YoloPaths.face_trained_weights_path)
model = YOLO("yolo11m.pt")

folder_name = "person_validation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
# Validate the model
metrics = model.val(data=str(YoloPaths.person_data_config_path),
                    save_json=True, 
                    iou=0.5, 
                    plots=True, 
                    project=str(YoloPaths.person_output_dir),
                    name=folder_name)
# Extract precision and recall
precision = metrics.results_dict['metrics/precision(B)']
recall = metrics.results_dict['metrics/recall(B)']
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Log results
logging.info(f"Precision: {precision:.4f}")
logging.info(f"Recall: {recall:.4f}")
logging.info(f"F1 Score: {f1_score:.4f}")

# save precision and recall to a file
with open(YoloPaths.person_output_dir / folder_name / "precision_recall.txt", "w") as f:
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1_score}\n")
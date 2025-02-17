import argparse
import logging
from datetime import datetime
from ultralytics import YOLO
from constants import YoloPaths

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model.")
    parser.add_argument(
        "--yolo_target", 
        type=str, 
        help="What yolo model to evaluate."
    )
    args = parser.parse_args()

    # Load the YOLO model with the supplied target weights
    if args.yolo_target == "person_face":
        model = YOLO(YoloPaths.person_face_trained_weights_path)
        folder_name = "person_validation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        data_config = str(YoloPaths.person_face_data_config_path)
        project_folder = str(YoloPaths.person_face_output_dir)
    elif args.yolo_target == "gaze":
        model = YOLO(YoloPaths.gaze_trained_weights_path)
        folder_name = "gaze_validation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        data_config = str(YoloPaths.gaze_data_config_path)
        project_folder = str(YoloPaths.gaze_output_dir)
    # Validate the model
    metrics = model.val(
        data=data_config,
        save_json=True,
        iou=0.5,
        plots=True,
        project=project_folder,
        name=folder_name
    )

    # Extract precision and recall
    precision = metrics.results_dict['metrics/precision(B)']
    recall = metrics.results_dict['metrics/recall(B)']
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Log results
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1_score:.4f}")

    # Save precision and recall to a file
    with open(YoloPaths.person_output_dir / folder_name / "precision_recall.txt", "w") as f:
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {f1_score}\n")

if __name__ == '__main__':
    main()
from ultralytics import YOLO
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from constants import YoloPaths

def save_plot(plt, filename):
    """Saves the plot to the specified filename."""
    plt.savefig(filename)
    print(f"Plot saved to: {filename}")
    
def plot_precision_recall_curve(true_labels, pred_scores, output_dir):
    """Plots the Precision-Recall Curve and saves it."""
    precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid()
    
    # Save plot
    save_plot(plt, os.path.join(output_dir, "precision_recall_curve.png"))

def plot_roc_curve(true_labels, pred_scores, output_dir):
    """Plots the ROC Curve and computes AUC, then saves it."""
    fpr, tpr, _ = roc_curve(true_labels, pred_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()

    # Save plot
    save_plot(plt, os.path.join(output_dir, "roc_curve.png"))
 
def evaluate_model(best_model_path, test_data_path, output_dir):
    """Evaluates the model and saves performance plots."""
    # Load the trained YOLO model
    model = YOLO(best_model_path)

    # Evaluate on the test dataset
    results = model.val(
        data=YoloPaths.face_data_config_path, 
        imgsz=640,
        batch=16
    )
    print("RESULTS:", results)
    # Retrieve predictions and ground truth from the results
    pred_scores = results.pred 
    true_labels = results.labels 

    # # Generate performance metrics
    # print(f"Test mAP@0.5: {results.metrics['mAP50']}")
    # print(f"Test mAP@0.5:0.95: {results.metrics['mAP50-95']}")

    # # Generate and save plots
    # plot_precision_recall_curve(true_labels, pred_scores, output_dir)
    # plot_roc_curve(true_labels, pred_scores, output_dir)
       
# Path to the best model weights and test dataset
best_model_path = str(YoloPaths.face_output_dir) + "/yolo_face_finetune7/weights/best.pt"
test_data_path = "/home/nele_pauline_suffo/ProcessedData/yolo_face_input/images/test"
output_dir = str(YoloPaths.face_output_dir) + "/yolo_face_finetune7"  # Use the same output directory

# Evaluate the model and generate plots
evaluate_model(best_model_path, test_data_path, output_dir)
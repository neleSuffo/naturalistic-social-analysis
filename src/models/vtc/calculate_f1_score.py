import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

def calculate_overlap(start_pred, end_pred, start_annot, end_annot):
    """
    Calculate the overlap duration between two time intervals.
    """
    overlap_start = max(start_pred, start_annot)
    overlap_end = min(end_pred, end_annot)
    return max(0, overlap_end - overlap_start)

def calculate_f1_score(predictions, annotations, output_file="metrics_results.txt"):
    """
    Calculate F1 score for multi-class duration-based metrics.
    Args:
        predictions (pd.DataFrame): Prediction dataframe with columns ['audio_file_name', 'start', 'end', 'voice_type'].
        annotations (pd.DataFrame): Ground truth dataframe with columns ['audio_file_name', 'start', 'end', 'voice_type'].
    Returns:
        dict: F1 scores per class and the macro F1 score.
    """
    # Ensure all unique voice types are included
    all_classes = set(predictions['Voice_type'].unique()) | set(annotations['Voice_type'].unique())
    results = {}

    for voice_type in all_classes:
        # Filter data for the current class
        pred_class = predictions[predictions['Voice_type'] == voice_type]
        annot_class = annotations[annotations['Voice_type'] == voice_type]

        tp_duration = 0.0
        fp_duration = 0.0
        fn_duration = 0.0

        # Calculate True Positives (TP) and False Positives (FP)
        for _, pred_row in pred_class.iterrows():
            pred_start = pred_row['Utterance_Start']
            pred_end = pred_row['Utterance_End']

            # Find overlap with all annotations for the same class
            overlap = annot_class.apply(
                lambda row: calculate_overlap(pred_start, pred_end, row['Utterance_Start'], row['Utterance_End']), axis=1
            )
            overlap_duration = overlap.sum()
            tp_duration += overlap_duration
            fp_duration += (pred_end - pred_start) - overlap_duration

        # Calculate False Negatives (FN)
        for _, annot_row in annot_class.iterrows():
            annot_start = annot_row['Utterance_Start']
            annot_end = annot_row['Utterance_End']

            # Find overlap with all predictions for the same class
            overlap = pred_class.apply(
                lambda row: calculate_overlap(annot_start, annot_end, row['Utterance_Start'], row['Utterance_End']), axis=1
            )
            overlap_duration = overlap.sum()
            fn_duration += (annot_end - annot_start) - overlap_duration

        # Compute precision, recall, F1 score
        precision = tp_duration / (tp_duration + fp_duration) if (tp_duration + fp_duration) > 0 else 0.0
        recall = tp_duration / (tp_duration + fn_duration) if (tp_duration + fn_duration) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        results[voice_type] = {'precision': precision, 'recall': recall, 'f1': f1}

    # Calculate Macro F1 Score
    f1_scores = [class_metrics['f1'] for class_metrics in results.values()]
    macro_f1 = sum(f1_scores) / len(f1_scores)

    # Save results to file
    with open(output_file, "w") as f:
        f.write("Class-wise Metrics:\n")
        for voice_type, metrics in results.items():
            f.write(f"{voice_type}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}\n")
        f.write(f"\nMacro F1 Score: {macro_f1:.3f}\n")

    return results, macro_f1


if __name__ == "__main__":
    vtc_share_output = pd.read_pickle('/home/nele_pauline_suffo/outputs/vtc/quantex_share_vtc_output.pkl')
    annotations_output = pd.read_pickle('/home/nele_pauline_suffo/ProcessedData/annotations_superannotate/quantex_share_annotations.pkl')
    
    # Calculate F1 Scores
    class_results, macro_f1 = calculate_f1_score(vtc_share_output, annotations_output)

    # Output Results (File and Terminal)
    print("Class-wise Metrics:")
    for voice_type, metrics in class_results.items():
        print(f"{voice_type}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
    print(f"\nMacro F1 Score: {macro_f1:.3f}")
import pandas as pd
from pyannote.metrics.detection import DetectionPrecisionRecallFMeasure
from pyannote.core import Annotation, Segment

def dataframe_to_annotation(df, label_column="Voice_type"):
    """
    Converts a DataFrame to a pyannote.core.Annotation object.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with 'Utterance_Start', 'Utterance_End', and a label column.
    - label_column (str): Column name for the labels (default: 'Voice_type').

    Returns:
    - Annotation: pyannote.core.Annotation object.
    """
    annotation = Annotation()
    for _, row in df.iterrows():
        start = float(row["Utterance_Start"])
        end = float(row["Utterance_End"])
        label = row[label_column]
        annotation[Segment(start, end)] = label
    return annotation

reference_df = pd.read_pickle('/home/nele_pauline_suffo/ProcessedData/annotations_superannotate/quantex_share_annotations.pkl')
hypothesis_df = pd.read_pickle('/home/nele_pauline_suffo/outputs/vtc/quantex_share_vtc_output.pkl')
reference = dataframe_to_annotation(reference_df)
hypothesis = dataframe_to_annotation(hypothesis_df)

# Initialize metric
detection_metric = DetectionPrecisionRecallFMeasure(collar=0, skip_overlap=False)

# Compute precision, recall, and F1 score
precision, recall, f1 = detection_metric(reference, hypothesis)

# Print results
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
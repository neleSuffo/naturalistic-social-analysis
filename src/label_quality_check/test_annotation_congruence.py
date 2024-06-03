import pandas as pd
import json
import re

# df_annotations = pd.read_parquet(
#     "/Users/nelesuffo/projects/leuphana-IPE/data/annotations/261609_005_annotation.parquet"
# )
# df_annotation_video = pd.read_parquet(
#     "/Users/nelesuffo/projects/leuphana-IPE/data/annotation_examples/261609_005_annotation.parquet"
# )

# # print(df_annotations.head())
# is_identical = df_annotations.equals(df_annotation_video)
# print(is_identical)


def process_annotations(file_path: str) -> pd.DataFrame:
    # Load JSON file
    with open(file_path) as f:
        data = json.load(f)

    # Extract the image id and the corresponding filename
    id_filename_dict = {
        item["id"]: re.findall(r"\d+", item["file_name"])[0] for item in data["images"]
    }

    # Convert annotations list to DataFrame
    annotations_df = pd.DataFrame(data["annotations"])

    # Add a column with the filename and rename it to 'frame'
    annotations_df["frame"] = annotations_df["image_id"].map(id_filename_dict)

    # Split 'bbox' column into four separate columns
    annotations_df[["xtl", "ytl", "xbr", "ybr"]] = pd.DataFrame(
        annotations_df["bbox"].tolist(), index=annotations_df.index
    )

    # Split 'attributes' column into separate columns
    annotations_df = annotations_df.join(annotations_df["attributes"].apply(pd.Series))

    # Reorder the columns to have 'frame' as the first column
    annotations_df = annotations_df[
        ["frame"] + [col for col in annotations_df.columns if col != "frame"]
    ]

    return annotations_df


# file_path = '/Users/nelesuffo/projects/leuphana-IPE/data/annotation_examples/instances_default.json'
file_path = (
    "/Users/nelesuffo/projects/leuphana-IPE/data/annotations/instances_default.json"
)
annotations_df = process_annotations(file_path)
print(annotations_df.head())

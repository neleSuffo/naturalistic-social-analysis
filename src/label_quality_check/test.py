import pandas as pd

df_annotations = pd.read_parquet(
    "/Users/nelesuffo/projects/leuphana-IPE/data/all_annotations/255237_004_2_annotation.parquet"
)
df_annotation_video = pd.read_parquet(
    "/Users/nelesuffo/projects/leuphana-IPE/data/annotations/255237_004_2_annotation.parquet"
)

# print(df_annotations.head())
is_identical = df_annotations.equals(df_annotation_video)
print(is_identical)

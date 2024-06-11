# Define constants for the annotation dataset
# irrelevant columns
drop_columns = ["source", "occluded", "z_order"]

# Map 'yes' and 'no' to 1 and 0 for boolean column 'Interaction'
interaction_map = {"yes": 1, "no": 0}

# Map age categories to integers
age_map = {"inf": 0, "child": 1, "teen": 2, "adult": 3, "dk": -1}

# The columns that need to be converted to strings for mapping
str_columns = ["Age", "label"]

# The columns that need to be converted to integers
int_columns = [
    "id",
    "label",
    "frame",
    "Visibility",
    "Interaction",
    "Age",
    "ID",
    "outside",
    "keyframe",
]

# The columns that need to be converted to floats
float_columns = ["xtl", "ytl", "xbr", "ybr"]


# Path variable to the annotation xml files
annotations_path = "data/annotations/"

# Path variable to the output json files
model_output_path = "output/"

# Description: Configuration file for the label quality check module

# Path to the annotations input file
annotations_input_path = (
    "/Users/nelesuffo/projects/leuphana-IPE/data/annotations/annotations.xml"
)
# Path to the annotations output directory
annotations_output_path = "/Users/nelesuffo/projects/leuphana-IPE/data/annotations/"

import fiftyone as fo


name = "task_261609_005-2024_04_08_12_03_04-cvat for video"
# The directory containing the dataset to import
dataset_dir = "/Users/nelesuffo/projects/leuphana-IPE/data/frame.MP4"

# The type of the dataset being imported
dataset_type = fo.types.CVATVideoDataset

dataset = fo.Dataset.from_videos_dir(dataset_dir)


session = fo.launch_app(dataset, port=5151)

session.wait()

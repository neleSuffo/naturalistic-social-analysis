import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
import subprocess
import sys
from constants import Inference
from utils import extract_child_id

def run_fold(fold_nr, test_videos, mode="tertiary"):
    """Calls existing pipeline with a temporary video list for this fold."""
    fold_dir = Path(f"cross_val_fold_{fold_nr}")
    fold_dir.mkdir(exist_ok=True)

    temp_list_path = fold_dir / "temp_test_list.txt"
    with open(temp_list_path, "w") as f:
        f.write("\n".join(test_videos))

    print(f"\n🚀 STARTING FOLD {fold_nr}/5")
    print(f"Testing on {len(test_videos)} videos from unique children.")

    cmd = [
        sys.executable,
        "pipeline_runner.py",
        "--mode",
        mode,
        "--plot",
    ]

    subprocess.run(cmd, check=True)


def main():
    # Load video names
    with open(Inference.QUANTEX_VIDEOS_LIST_FILE, "r") as f:
        videos = [line.strip() for line in f if line.strip()]

    # Build metadata dataframe
    video_metadata = pd.DataFrame({"video_name": videos})
    video_metadata["child_id"] = video_metadata["video_name"].apply(
        lambda x: extract_child_id(x) or "unknown"
    )

    # Group-based CV by child
    gkf = GroupKFold(n_splits=5)

    for i, (_, test_idx) in enumerate(
        gkf.split(video_metadata, groups=video_metadata["child_id"])
    ):
        test_videos = video_metadata.iloc[test_idx]["video_name"].tolist()
        run_fold(i + 1, test_videos)


if __name__ == "__main__":
    main()
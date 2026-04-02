import subprocess
import sys
import re
import argparse
import logging
from pathlib import Path
from constants import Analysis, Inference

# --- Configuration: Ensure these paths match your folder structure ---
FRAME_ANALYSIS_SCRIPT = Path("analysis/pipeline_frame_level_analysis.py")
SEGMENT_CREATION_SCRIPT = Path("analysis/pipeline_video_level_analysis.py")
EVALUATION_SCRIPT = Path("analysis/validation/eval_segment_performance.py")

def run_command(cmd, step_name):
    """Executes a subprocess command and handles output stream logging."""
    print(f"\n--- Starting: {step_name} ---")
    try:
        # We use capture_output=False here so you can see the 'Smoothing...' 
        # and '✅' logs in real-time in your console.
        result = subprocess.run(
            cmd,
            check=True,
            cwd=Path(__file__).parent.parent,
            text=True,
            encoding='utf-8'
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR in {step_name}: Command failed.")
        sys.exit(1)

def main(social_state_mode='tertiary', plot=False, video_list=None):
    """
    Runs the streamlined inference pipeline.
    
    Parameters:
    ----------
    social_state_mode (str): 
        'binary' or 'tertiary' classification mode. Default is 'tertiary'.
    plot (bool): 
        Whether to generate timeline comparison plots (requires GT). Default is False.
    video_list (list of str): 
        Optional list of specific video filenames to process (e.g. ['video1.mp4', 'video2.mp4']).
    """
    # 1. --- STEP 1: FRAME-LEVEL ANALYSIS ---
    # This generates the frame_level_social_interactions.csv
    frame_cmd = [
        sys.executable, str(FRAME_ANALYSIS_SCRIPT),
        "--social_state_mode", social_state_mode
    ]
    if video_list:
        frame_cmd.extend(["--video_list"] + video_list)

    run_command(frame_cmd, "Step 1: Frame-Level Analysis")

    # The frame-level script saves to a timestamped folder inside Analysis.BASE_OUTPUT_DIR
    # We find the most recent one to continue the pipeline
    all_runs = sorted(Analysis.BASE_OUTPUT_DIR.glob("analysis_*"))
    if not all_runs:
        print("❌ ERROR: No analysis output folder found.")
        sys.exit(1)
    
    run_folder = all_runs[-1]
    print(f"➡️ Processing Results in: {run_folder}")

    # 2. --- STEP 2: SEGMENT CREATION (Smoothing & Merging) ---
    segment_cmd = [
        sys.executable, str(SEGMENT_CREATION_SCRIPT),
        "--output_folder_path", str(run_folder),
        "--social_state_mode", social_state_mode
    ]
    run_command(segment_cmd, "Step 2: Video-Level Smoothing")

    # 3. --- STEP 3: EVALUATION (Optional) ---
    # Only runs if Ground Truth is available for the videos in video_list
    evaluation_cmd = [
        sys.executable, str(EVALUATION_SCRIPT),
        "--folder_path", str(run_folder),
        "--social_state_mode", social_state_mode
    ]
    if plot:
        evaluation_cmd.append("--plot")
    if video_list:
        evaluation_cmd.extend(["--video_list"] + video_list)
        
    run_command(evaluation_cmd, "Step 3: Evaluation and Plotting")
    
    print("\n" + "="*40)
    print(f"🎉 INFERENCE COMPLETE")
    print(f"Final segments saved: {run_folder / Analysis.INTERACTION_SEGMENTS_CSV.name}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Social Analysis Inference Runner")
    parser.add_argument("--social_state_mode", type=str, choices=['binary', 'tertiary'], default='tertiary',
                        help="Classification mode (default: tertiary)")
    parser.add_argument("--plot", action='store_true', 
                        help="Generate timeline comparison plots (requires GT)")
    parser.add_argument("--video_list", type=str, nargs='+', 
                        help="List of specific videos to process (e.g. video1.mp4 video2.mp4)")
    
    args = parser.parse_args()
    
    # check if args.video_list is a txt file containing video names o
    if args.video_list and len(args.video_list) == 1 and args.video_list[0].endswith('.txt'):
        with open(args.video_list[0], 'r') as f:
            video_list = [line.strip() for line in f if line.strip()]
        args.video_list = video_list
    # check if args.video_list is a folder containing videos
    elif args.video_list and len(args.video_list) == 1 and Path(args.video_list[0]).is_dir():
        video_folder = Path(args.video_list[0])
        video_list = [str(video) for video in video_folder.iterdir() if video.is_file() and video.suffix.lower() == ".mp4"]
        args.video_list = video_list
    main(social_state_mode=args.social_state_mode, plot=args.plot, video_list=args.video_list)
import subprocess
import sys
import logging
import argparse
from pathlib import Path
from constants import Analysis

def run_command(cmd: list, 
                step_name: str) -> bool:
    """
    Executes a subprocess command and handles output stream logging.
    
    Parameters:
    ----------
    cmd: list
        The command to execute as a list of strings.
    step_name: str
        A descriptive name for the step being executed (used in logs).

    Returns
    -------
    bool
        True if the command executed successfully, False otherwise.
    """
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

def main(social_state_mode='tertiary', 
         video_list=None):
    """
    Runs the streamlined inference pipeline.
    
    Parameters:
    ----------
    social_state_mode (str): 
        'binary' or 'tertiary' classification mode. Default is 'tertiary'.
    video_list (list of str): 
        Optional list of specific video filenames to process (e.g. ['video1.mp4', 'video2.mp4']).
    """
    # 1. --- STEP 1: FRAME-LEVEL ANALYSIS ---
    frame_cmd = [
        sys.executable, str(Analysis.FRAME_ANALYSIS_SCRIPT),
        "--social_state_mode", social_state_mode
    ]
    if video_list:
        frame_cmd.extend(["--video_list"] + video_list)

    run_command(frame_cmd, "Step 1: Frame-Level Analysis")
    
    # 2. --- IDENTIFY OUTPUT FOLDER ---
    # Sort by modification time to be sure we get the one JUST created
    all_runs = sorted(Analysis.BASE_OUTPUT_DIR.glob("analysis_*"), key=lambda x: x.stat().st_mtime)
    if not all_runs:
        logging.error("❌ ERROR: No analysis output folder found.")
        sys.exit(1)
    
    run_folder = all_runs[-1]
    logging.info(f"➡️ Processing Results in: {run_folder}")

    # 3. --- STEP 2: SEGMENT CREATION ---
    segment_cmd = [
        sys.executable, str(Analysis.SEGMENT_CREATION_SCRIPT),
        "--output_folder_path", str(run_folder),
        "--social_state_mode", social_state_mode
    ]
    run_command(segment_cmd, "Step 2: Video-Level Smoothing")
    
    logging.info("\n" + "="*40)
    logging.info(f"🎉 INFERENCE COMPLETE")
    logging.info(f"Final segments saved: {run_folder / Analysis.INTERACTION_SEGMENTS_CSV.name}")
    logging.info("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Social Analysis Inference Runner")
    parser.add_argument("--social_state_mode", type=str, choices=['binary', 'tertiary'], default='tertiary',
                        help="Classification mode (default: tertiary)")
    parser.add_argument("--video_list", type=str, nargs='+', 
                        help="List of specific videos to process (e.g. video1.mp4 video2.mp4)")
    
    args = parser.parse_args()
    
    # check if args.video_list is a txt file containing video names or a folder containing videos
    if args.video_list and len(args.video_list) == 1 and args.video_list[0].endswith('.txt'):
        with open(args.video_list[0], 'r') as f:
            video_list = [line.strip() for line in f if line.strip()]
        args.video_list = video_list
    # check if args.video_list is a folder containing videos
    elif args.video_list and len(args.video_list) == 1 and Path(args.video_list[0]).is_dir():
        video_folder = Path(args.video_list[0])
        video_list = [str(video) for video in video_folder.iterdir() if video.is_file() and video.suffix.lower() == ".mp4"]
        args.video_list = video_list
    main(social_state_mode=args.social_state_mode, video_list=args.video_list)
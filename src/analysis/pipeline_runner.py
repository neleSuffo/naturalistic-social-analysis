import subprocess
import sys
import re
import argparse
from pathlib import Path
from datetime import datetime
from constants import Inference

# --- Configuration ---
FRAME_ANALYSIS_SCRIPT = Path("analysis/pipeline_frame_level_analysis.py")
SEGMENT_CREATION_SCRIPT = Path("analysis/pipeline_video_level_analysis.py")
EVALUATION_SCRIPT = Path("analysis/validation/eval_segment_performance.py")

def run_command(cmd, step_name):
    """Executes a subprocess command and handles errors."""
    print(f"\n--- Starting: {step_name} ---")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            encoding='utf-8'
        )
        print(f"✅ {step_name} completed successfully.")
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR in {step_name}: Command failed with return code {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"❌ ERROR: Python or script not found.")
        sys.exit(1)

def main(rules=None, plot=False, mode='tertiary', video_list=None):
    """
    Runs the full pipeline.
    """    
    # 1. --- STEP 1: FRAME-LEVEL ANALYSIS ---
    frame_cmd = [
        sys.executable, 
        str(FRAME_ANALYSIS_SCRIPT),
        "--mode", mode
    ]
    if rules:
        frame_cmd.extend(["--rules"] + [str(r) for r in rules])
    
    # --- NEW: Pass the video list if provided ---
    if video_list:
        # We pass the videos as a space-separated string or a path to a temp file
        # Assuming your frame_level script is updated to accept --video_list
        frame_cmd.extend(["--video_list"] + video_list)
    
    frame_output = run_command(frame_cmd, "Frame-Level Analysis")
    
    path_match = re.search(r'interaction_analysis_\d{8}_\d{6}', frame_output)
    
    if path_match:
        run_folder = Inference.BASE_OUTPUT_DIR / path_match.group(0)
    else:
        print(f"❌ ERROR: Folder path not found. Output:\n{frame_output}")
        sys.exit(1)
    
    print(f"➡️ Captured Run Folder: {run_folder}")

    # 2. --- STEP 2: SEGMENT CREATION ---
    segment_cmd = [
        sys.executable, 
        str(SEGMENT_CREATION_SCRIPT),
        "--folder_path", str(run_folder),
        "--mode", mode
    ]
    run_command(segment_cmd, "Segment Creation")

    # 3. --- STEP 3: EVALUATION ---    
    evaluation_cmd = [
        sys.executable, 
        str(EVALUATION_SCRIPT),
        "--folder_path", str(run_folder),
        "--mode", mode
    ]
    if plot:
        evaluation_cmd.append("--plot")
    if video_list:
        evaluation_cmd.extend(["--video_list"] + video_list)
        
    run_command(evaluation_cmd, "Evaluation and Plotting")
    
    print("\n\n🎉 Full Pipeline Execution Complete!")
    print(f"Final results saved in: {run_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full multimodal analysis pipeline.")
    parser.add_argument("--rules", type=int, nargs='+', help="Override default rule set.")
    parser.add_argument("--mode", type=str, choices=['binary', 'tertiary'], default='tertiary')
    parser.add_argument("--plot", action='store_true')
    
    # --- NEW: Argument for Cross-Validation script to use ---
    parser.add_argument("--video_list", type=str, nargs='+', help="List of specific videos to process.")
    
    args = parser.parse_args()
    
    main(rules=args.rules, plot=args.plot, mode=args.mode, video_list=args.video_list)
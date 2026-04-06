"""
Hyperparameter Tuning for Social Interaction Analysis

This script systematically tests different combinations of hyperparameters
for the social interaction analysis pipeline and evaluates their performance
against ground truth data. It supports selective processing of video subsets
to facilitate cross-validation and prevent data leakage.
"""

import pandas as pd
import sys
import random
import json
import time
import argparse 
from datetime import datetime
from pathlib import Path
from itertools import product

# Add the src directory to path for imports
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))

from constants import DataPaths, Analysis, Inference
from config import AnalysisConfig, HyperparameterConfig
from analysis.validation.eval_segment_performance import run_evaluation
from analysis.pipeline_frame_level_analysis import main as frame_analysis_main
from analysis.pipeline_video_level_analysis import main as segment_analysis_main

import random

def generate_hyperparameter_combinations(max_combinations=20, random_sample=True):
    """
    Memory-efficient hyperparameter generation. 
    Avoids Cartesian product to prevent OOM errors.
    """
    ranges = HyperparameterConfig.HYPERPARAMETER_RANGES
    param_names = list(ranges.keys())
    
    # If you aren't doing a random sample and want the full grid, 
    # only then should you use the product (and only if the grid is small).
    if not random_sample:
        # Standard Cartesian product (only use if you know the grid is small!)
        from itertools import product
        all_combos = [dict(zip(param_names, c)) for c in product(*[ranges[p] for p in param_names])]
        return all_combos[:max_combinations] if max_combinations else all_combos

    # --- MEMORY EFFICIENT RANDOM SEARCH ---
    combinations = []
    seen = set() # To ensure uniqueness
    
    # Safety break to avoid infinite loops if max_combos > possible combos
    total_possible = 1
    for r in ranges.values(): total_possible *= len(r)
    limit = min(max_combinations, total_possible)

    while len(combinations) < limit:
        # Pick one random value for every parameter
        combo = {p: random.choice(ranges[p]) for p in param_names}
        
        # Convert to a frozenset of items to make it hashable for the 'seen' check
        combo_frozen = tuple(sorted(combo.items()))
        
        if combo_frozen not in seen:
            seen.add(combo_frozen)
            combinations.append(combo)
            
    return combinations

def run_pipeline_for_combo(hyperparameters, combo_dir, hyperparameter_tuning, social_state_mode,video_list=None):
    """
    Runs the full analysis pipeline by temporarily setting AnalysisConfig 
    attributes and executing the analysis functions.
    
    Parameters
    ----------
    hyperparameters: dict
        Hyperparameters to set for this run.
    combo_dir: Path
        Directory where outputs for this combination will be stored.
    hyperparameter_tuning: bool
        Whether to enable hyperparameter tuning.
    social_state_mode: str
        tertiary or binary evaluation mode (affects evaluation metrics and thresholds)
    video_list: list, optional
        Specific videos to process. If provided, Step 1 will only query these videos.
    """
    original_config = {}

    try:
        # 1. Temporarily override AnalysisConfig attributes
        for key, value in hyperparameters.items():
            if hasattr(AnalysisConfig, key):
                original_config[key] = getattr(AnalysisConfig, key)
                setattr(AnalysisConfig, key, value)
                
        frame_output_path = combo_dir / Analysis.FRAME_LEVEL_INTERACTIONS_CSV.name
        segment_output_path = combo_dir / Analysis.INTERACTION_SEGMENTS_CSV.name
        
        # 2. Execute Frame-Level Analysis (Passing the video filter)
        frame_analysis_main(
            db_path=DataPaths.INFERENCE_DB_PATH,
            output_dir=combo_dir,
            hyperparameter_tuning=hyperparameter_tuning,
            video_list=video_list,
            social_state_mode=social_state_mode
        )

        # 3. Execute Segment-Level Analysis
        segment_analysis_main(
            output_file_path=segment_output_path, 
            frame_data_path=frame_output_path,
            hyperparameter_tuning=hyperparameter_tuning,
            social_state_mode=social_state_mode
        )
        
        return True, frame_output_path, segment_output_path, None

    except Exception as e:
        return False, None, None, f"Pipeline execution failed: {str(e)}"
        
    finally:
        # 4. Reset AnalysisConfig to original state
        for key, value in original_config.items():
            setattr(AnalysisConfig, key, value)

def run_analysis_with_config(hyperparameters, combo_id, output_base_dir, hyperparameter_tuning, social_state_mode, video_list=None):
    """
    Wrapper function to manage directory creation and pipeline execution for a combination.
    
    Parameters
    ----------
    hyperparameters: dict
        Hyperparameters to set for this run.
    combo_id: int
        Unique identifier for this combination (used for directory naming).
    output_base_dir: Path
        Base directory where results for this combination will be stored.
    hyperparameter_tuning: bool
        Whether to enable hyperparameter tuning.
    social_state_mode: str
        tertiary or binary evaluation mode (affects evaluation metrics and thresholds)
    video_list: list, optional
        Specific videos to process. If provided, Step 1 will only query these videos.
        
    Returns
    -------
    success: bool
        Whether the pipeline ran successfully.
    frame_output_path: Path or None
        Path to frame-level output CSV if successful, else None.
    segment_output_path: Path or None
        Path to segment-level output CSV if successful, else None.
    error: str or None
        Error message if failed, else None.
    """
    combo_dir = output_base_dir / f"combo_{combo_id:04d}"
    combo_dir.mkdir(exist_ok=True, parents=True)
    
    with open(combo_dir / "hyperparameters.json", 'w') as f:
        json.dump(hyperparameters, f, indent=2)
        
    # Run the pipeline with the specified hyperparameters and video filter
    success, frame_out, seg_out, error = run_pipeline_for_combo(
        hyperparameters, combo_dir, hyperparameter_tuning, social_state_mode, video_list=video_list
    )
        
    return success, frame_out, seg_out, error

def evaluate_combination(segment_output_path, video_list=None):
    """
    Evaluates the performance of a specific hyperparameter combination by comparing
    the generated segments against ground truth annotations for the specified videos.
    
    Parameters    
    ----------
    segment_output_path: Path
        Path to the CSV file containing the predicted interaction segments for this combination.
    video_list: list, optional
        Specific videos to evaluate. If None, evaluates on all videos in the segment_output_path.
        
    Returns
    -------
    dict containing:
        success: bool
            Whether the evaluation ran successfully.
        overall_metrics: dict or None
            Dictionary of overall performance metrics (e.g., macro_avg_f1_score) if successful, else None.
        error: str or None
            Error message if evaluation failed, else None.
    """
    try:
        _, _, detailed_metrics = run_evaluation(
            predictions_path=segment_output_path, 
            output_folder=segment_output_path.parent,
            mode='tertiary', 
            video_list=video_list
        )

        if detailed_metrics and 'macro_avg' in detailed_metrics:
            overall_metrics = {
                'macro_avg_f1_score': detailed_metrics['macro_avg']['f1_score'],
                'macro_avg_precision': detailed_metrics['macro_avg']['precision'],
                'macro_avg_recall': detailed_metrics['macro_avg']['recall'],
            }
            return {'success': True, 'overall_metrics': overall_metrics, 'error': None}
        return {'success': False, 'overall_metrics': None, 'error': "No metrics"}
    except Exception as e:
        return {'success': False, 'overall_metrics': None, 'error': str(e)}

def find_best_configuration(results):
    """Finds the configuration with the highest Macro F1 score."""
    return max(results, key=lambda x: x['evaluation']['overall_metrics']['macro_f1'])

def main(max_combinations=None, video_list=None): 
    """Main hyperparameter tuning loop."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = Path(f"{Inference.HYPERPARAMETER_OUTPUT_DIR}_{timestamp}")
    output_base_dir.mkdir(exist_ok=True, parents=True)
    
    combinations = generate_hyperparameter_combinations(
        max_combinations=max_combinations, 
        random_sample=getattr(AnalysisConfig, 'RANDOM_SAMPLING', False)
    )
    
    all_results = []
    start_time = time.time()
    
    for i, hyperparams in enumerate(combinations):
        combo_id = i + 1
        print(f"\n[{combo_id}/{len(combinations)}] Testing Combo {combo_id}")
        
        # Run Analysis
        success, _, seg_out, error = run_analysis_with_config(
            hyperparams, combo_id, output_base_dir, video_list=video_list
        )
        
        if not success:
            print(f"❌ Failed: {error}")
            continue
        
        # Evaluate performance for the specific video subset
        eval_res = evaluate_combination(seg_out, video_list=video_list)
        
        if eval_res['success']:
            result_record = {
                'combo_id': combo_id,
                'hyperparameters': hyperparams,
                'evaluation': eval_res
            }
            all_results.append(result_record)
            print(f"✅ Macro F1: {eval_res['overall_metrics']['macro_f1']:.4f}")

    # Final Summaries
    if all_results:
        best_config = find_best_configuration(all_results)
        save_final_results(all_results, best_config, output_base_dir)
        print_results_summary(all_results, best_config)

def save_final_results(all_results, best_config, output_dir):
    """Saves result summary CSV and best configuration JSON."""
    summary_data = []
    for res in all_results:
        row = {
            'combo_id': res['combo_id'],
            'macro_f1': res['evaluation']['overall_metrics']['macro_f1'],
            **res['hyperparameters']
        }
        summary_data.append(row)
    
    pd.DataFrame(summary_data).to_csv(output_dir / "results_summary.csv", index=False)
    with open(output_dir / "best_configuration.json", 'w') as f:
        json.dump(best_config, f, indent=2)

def print_results_summary(all_results, best_config):
    """Prints best performance metrics to console."""
    print("\n" + "=" * 30 + "\n🏆 WINNER: Combo", best_config['combo_id'])
    for k, v in best_config['hyperparameters'].items():
        print(f"  {k}: {v}")
    print(f"  F1: {best_config['evaluation']['overall_metrics']['macro_f1']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-combos', type=int, default=20)
    parser.add_argument('--video_list', type=str, nargs='+', default=None)
    args = parser.parse_args()
    
    main(max_combinations=args.max_combos, video_list=args.video_list)
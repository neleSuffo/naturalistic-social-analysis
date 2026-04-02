import argparse
import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupKFold
from datetime import datetime
from tune_hyperparameters import generate_hyperparameter_combinations, run_analysis_with_config, evaluate_combination
from constants import Analysis
from config import AnalysisConfig
from utils import extract_child_id

def get_folds():
    """
    Splits all available annotated videos into 5 groups by Child ID.
    
    Returns:
    --------
    folds: list of tuples
        List of (train_indices, test_indices) for each fold
    df: pandas DataFrame
        DataFrame containing video names and corresponding Child IDs
    """
    # Load video list and extract Child IDs
    with open(Analysis.QUANTEX_VIDEOS_LIST_FILE, "r") as f:
        videos = [line.strip() for line in f if line.strip()]

    df = pd.DataFrame({"video_name": videos})
    df["child_id"] = df["video_name"].apply(lambda x: extract_child_id(x) or "unknown")
    
    # Split into folds using GroupKFold to ensure all videos of the same child are in the same fold
    gkf = GroupKFold(n_splits=AnalysisConfig.NUM_FOLDS)
    return list(gkf.split(df, groups=df['child_id'])), df

def run_cross_validation(mode="validation", social_state_mode="tertiary", max_combos=20):
    """
    Runs NUM_FOLDS-fold cross-validation for the analysis pipeline.
    
    Parameters:
    -----------
    mode: str
        "validation" for standard CV, "tuning" for hyperparameter tuning (evaluates on a smaller subset of combinations)
    social_state_mode: str
        "tertiary" or "binary" evaluation mode (affects evaluation metrics and thresholds)
    max_combos: int
        Maximum number of hyperparameter combinations to evaluate during tuning (only applicable in "tuning"
    """
    folds, df = get_folds()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Analysis.BASE_OUTPUT_DIR / f"cv_{mode}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    fold_results = []

    # --- CROSS-VALIDATION LOOP ---
    for i, (train_idx, test_idx) in enumerate(folds):
        fold_id = i + 1
        train_videos = df.iloc[train_idx]['video_name'].tolist()
        test_videos = df.iloc[test_idx]['video_name'].tolist()
        
        print(f"\n🚀 FOLD {fold_id}/{AnalysisConfig.NUM_FOLDS}")

        if mode == "tuning":
            # (Tuning logic removed for brevity, but ensure it uses the same fix as below)
            pass
        else:
            print(f"📊 Validating on: {len(test_videos)} videos")
            
            # Use ALL params from AnalysisConfig
            current_params = {
                item: getattr(AnalysisConfig, item) 
                for item in dir(AnalysisConfig) 
                if not item.startswith("__") and not callable(getattr(AnalysisConfig, item))
            }
            
            # 1. Run the pipeline
            success, _, seg_path, error = run_analysis_with_config(
                current_params, 
                fold_id, 
                output_root, 
                hyperparameter_tuning=False,
                social_state_mode=social_state_mode,
                video_list=test_videos
            )
            
            if success:
                # 2. Evaluate
                res = evaluate_combination(seg_path, video_list=test_videos)
                if res['success']:
                    fold_results.append(res['overall_metrics'])
                else:
                    print(f"⚠️ Eval failed for fold {fold_id}: {res['error']}")
            else:
                print(f"❌ Pipeline failed for fold {fold_id}: {error}")
            
    # --- AGGREGATE RESULTS ---
    if not fold_results:
        print("❌ No results collected. Check if database query returned 0 rows.")
        return

    results_df = pd.DataFrame(fold_results)
    print("\n" + "="*40 + "\n🏁 SUMMARY\n" + "="*40)
    
    # Safety check for describe
    stats = results_df.describe()
    if not stats.empty and 'mean' in stats.index:
        print(stats.loc[['mean', 'std']])
    else:
        print(results_df)
        
    results_df.to_csv(output_root / "cv_summary.csv")
    
    # save copy of pipeline_frame_level_analysis.py and pipeline_video_level_analysis.py for reference
    shutil.copy("pipeline_frame_level_analysis.py", output_root / "pipeline_frame_level_analysis.py")
    shutil.copy("pipeline_video_level_analysis.py", output_root / "pipeline_video_level_analysis.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_combos', type=int, default=20, help='Maximum number of hyperparameter combinations to evaluate during tuning (only applicable in "tuning" mode)')
    parser.add_argument('--social_state_mode', type=str, default='tertiary', help='Evaluation mode for social state analysis')
    args = parser.parse_args()
    
    run_cross_validation(mode="validation", social_state_mode=args.social_state_mode, max_combos=args.max_combos)
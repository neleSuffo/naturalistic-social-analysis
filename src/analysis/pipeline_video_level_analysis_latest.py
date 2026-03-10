import re
import argparse
import sys
import pandas as pd
import shutil
import ruptures as rpt
import numpy as np
from pathlib import Path

# Get the src directory (2 levels up from current notebook location)
src_path = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent
sys.path.append(str(src_path))

from constants import Inference, Evaluation, DataPaths
from config import DataConfig, InferenceConfig

# Constants
FPS = DataConfig.FPS # frames per second

def get_min_segment_duration(interaction_type: str) -> float:
    """
    Returns the minimum required segment duration (in seconds) based on the interaction type.
    
    This function relies on the assumption that the following constants are defined
    in InferenceConfig:
    - MIN_INTERACTING_SEGMENT_DURATION_SEC
    - MIN_ALONE_SEGMENT_DURATION_SEC
    - MIN_AVAILABLE_SEGMENT_DURATION_SEC
    - MIN_NOT_INTERACTING_SEGMENT_DURATION_SEC
    
    Parameters
    ----------
    interaction_type : str
        The label of the interaction state.
        
    Returns
    -------
    float
        Minimum duration in seconds.
    """
    type_map = {
        'Interacting': InferenceConfig.MIN_INTERACTING_SEGMENT_DURATION_SEC,
        'Alone': InferenceConfig.MIN_ALONE_SEGMENT_DURATION_SEC,
        'Available': InferenceConfig.MIN_AVAILABLE_SEGMENT_DURATION_SEC,
        'Not Interacting': InferenceConfig.MIN_NOT_INTERACTING_SEGMENT_DURATION_SEC,
    }
    default_min = getattr(InferenceConfig, 'MIN_ALONE_SEGMENT_DURATION_SEC', 15.0) 
    return type_map.get(interaction_type, default_min)

def apply_cpd_smoothing(frame_data: pd.DataFrame, mode: str):
    """
    Applies Change Point Detection (CPD) smoothing to the frame-level data.
    Uses the PELT algorithm to identify regime changes based on presence scores.

    Parameters
    ----------
    frame_data : pd.DataFrame
        The input DataFrame containing frame-level multimodal flags and presence scores.
    mode : str
        Classification mode: "binary" for Interacting vs Not Interacting, 
        "tertiary" for Interacting, Available, Alone.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with a smoothed 'interaction_type' column.
    """
    smoothed_results = []
    # PELT (Pruned Exact Linear Time) identifies the most likely regime changes 
    for _, video_df in frame_data.groupby('video_id'):
        video_df = video_df.sort_values('frame_number').copy()
        signal = video_df['presence_score'].values.reshape(-1, 1)
        
        algo = rpt.Pelt(model="l2").fit(signal)
        breakpoints = algo.predict(pen=InferenceConfig.CPD_PENALTY)

        start_idx = 0
        for end_idx in breakpoints:
            segment_slice = video_df.iloc[start_idx:end_idx]
            if not segment_slice.empty:
                counts = segment_slice['interaction_type'].value_counts(normalize=True)
                has_eng = (segment_slice['rule1_turn_taking'].any() | 
                           segment_slice['rule2_close_proximity'].any() | 
                           segment_slice['rule3_kcds_speaking'].any())
                
                if mode == "binary":
                    # Logic: If high-level rules triggered or density meets threshold
                    state = 1 if (has_eng or counts.get(1, 0) >= InferenceConfig.CPD_INTERACTING_THRESHOLD) else 2
                else:
                    # Hierarchical tertiary consensus logic
                    if has_eng and counts.get(1, 0) >= InferenceConfig.CPD_INTERACTING_THRESHOLD_LOW:
                        state = 1
                    elif counts.get(1, 0) >= InferenceConfig.CPD_INTERACTING_THRESHOLD:
                        state = 1
                    elif (counts.get(1, 0) + counts.get(2, 0)) >= InferenceConfig.CPD_TOTAL_PRESENCE_FLOOR:
                        state = 2
                    else:
                        state = 3
                
                video_df.iloc[start_idx:end_idx, video_df.columns.get_loc('interaction_type')] = state
            start_idx = end_idx
        smoothed_results.append(video_df)
        
    return pd.concat(smoothed_results)

def create_segments_for_video(video_id, video_df):
    """
    Create segments for a single video. Enforces type-specific minimum durations.
    
    Parameters
    ----------
    video_id : int
        Video identifier.
    video_df : pd.DataFrame
        Frame-level data for this video.
        
    Returns
    -------
    list
        List of segment dictionaries containing start/end frames and times.
    """
    video_df = video_df.sort_values('frame_number').reset_index(drop=True)
    if len(video_df) == 0:
        return []
    
    video_name = video_df['video_name'].iloc[0] 
    states = video_df['interaction_type'].values
    frame_nums = video_df['frame_number'].values
    segments = []
    
    curr_state = states[0]
    start_fr = frame_nums[0]
    
    for i in range(1, len(states)):
        if states[i] != curr_state:
            end_fr = frame_nums[i-1]
            dur = (end_fr - start_fr) / FPS
            
            if dur >= get_min_segment_duration(curr_state):
                segments.append({
                    'video_id': video_id, 'video_name': video_name,
                    'interaction_type': curr_state, 'segment_start': start_fr,
                    'segment_end': end_fr, 'start_time_sec': start_fr / FPS, 
                    'end_time_sec': end_fr / FPS, 'duration_sec': dur
                })
            start_fr = frame_nums[i]
            curr_state = states[i]
            
    # Final segment handler
    end_fr = frame_nums[-1]
    dur = (end_fr - start_fr) / FPS
    if dur >= get_min_segment_duration(curr_state):
        segments.append({
            'video_id': video_id, 'video_name': video_name,
            'interaction_type': curr_state, 'segment_start': start_fr,
            'segment_end': end_fr, 'start_time_sec': start_fr / FPS, 
            'end_time_sec': end_fr / FPS, 'duration_sec': dur
        })
    return segments

def merge_same_segments(segments_df, max_gap_sec=0.1):
    """
    Merge segments of the same category that have small gaps between them.
    
    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame with segments.
    max_gap_sec : float
        Maximum time gap (seconds) allowed to merge two segments of the same type.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with merged segments.
    """
    merged_segments = []
    for video_id, video_segments in segments_df.groupby('video_id'):
        video_segments = video_segments.sort_values('start_time_sec').reset_index(drop=True)
        if len(video_segments) == 0: continue
        
        current_segment = video_segments.iloc[0].copy()
        for i in range(1, len(video_segments)):
            next_segment = video_segments.iloc[i]
            gap_duration = next_segment['start_time_sec'] - current_segment['end_time_sec']
            
            # Merge if same type AND gap is negligible
            if (current_segment['interaction_type'] == next_segment['interaction_type'] and 
                gap_duration <= max_gap_sec):
                current_segment['segment_end'] = next_segment['segment_end']
                current_segment['end_time_sec'] = next_segment['end_time_sec']
                current_segment['duration_sec'] = current_segment['end_time_sec'] - current_segment['start_time_sec']
            else:
                merged_segments.append(current_segment.to_dict())
                current_segment = next_segment.copy()
        merged_segments.append(current_segment.to_dict())
        
    return pd.DataFrame(merged_segments) if merged_segments else segments_df

def fill_gaps_with_default(segments_df, default_type="Alone"):
    """
    Fills timeline gaps by either stretching segments or inserting default labels.
    
    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame with interaction segments.
    default_type : str
        The label to use for filling large gaps.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with a continuous timeline.
    """
    filled_segments = []
    for video_id, video_df in segments_df.groupby('video_id'):
        v_segs = video_df.sort_values('start_time_sec').to_dict('records')
        for i in range(len(v_segs)):
            filled_segments.append(v_segs[i])
            if i < len(v_segs) - 1:
                gap = v_segs[i+1]['start_time_sec'] - v_segs[i]['end_time_sec']
                if 0 < gap <= InferenceConfig.GAP_STRETCH_THRESHOLD:
                    filled_segments[-1]['end_time_sec'] = v_segs[i+1]['start_time_sec']
                    filled_segments[-1]['segment_end'] = v_segs[i+1]['segment_start'] - 1
                elif gap > InferenceConfig.GAP_STRETCH_THRESHOLD:
                    filled_segments.append({
                        'video_id': video_id, 'video_name': v_segs[i]['video_name'],
                        'interaction_type': default_type, 'start_time_sec': v_segs[i]['end_time_sec'],
                        'end_time_sec': v_segs[i+1]['start_time_sec'], 
                        'segment_start': v_segs[i]['segment_end'] + 1,
                        'segment_end': v_segs[i+1]['segment_start'] - 1
                    })
    return pd.DataFrame(filled_segments)

def print_segment_summary(segments_df, mode):
    """
    Print detailed summary statistics (minutes and percentages) for segments.
    
    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame with segments.
    mode : str
        "binary" or "tertiary".
    """
    if len(segments_df) > 0:
        total_min = round(segments_df['duration_sec'].sum() / 60, 2)
        print(f"\n📊 Final segment summary: {total_min} minutes total.")
        target_classes = ['Interacting', 'Not Interacting'] if mode == "binary" else ['Interacting', 'Alone', 'Available']
        for itype in target_classes:
            df_sub = segments_df[segments_df['interaction_type'] == itype]
            mins = round(df_sub['duration_sec'].sum() / 60, 2)
            perc = (mins / total_min * 100) if total_min > 0 else 0
            print(f"   {itype}: {len(df_sub)} segments ({mins}m - {perc:.1f}%)")
    else:
        print("\n📊 No segments created")

def main(output_file_path: Path, frame_data_path: Path, hyperparameter_tuning: bool = False, mode: str = "tertiary"):
    """
    Main entry point for segment analysis. Loads data, smooths, segments, and saves.
    """
    if hyperparameter_tuning:
        run_dir = output_file_path.parent
        try:
            shutil.copy(Path(__file__), run_dir / Path(__file__).name)
        except Exception: pass
        
    frame_data = pd.read_csv(frame_data_path)
    print("Smoothing frame-level data with CPD...")
    frame_data = apply_cpd_smoothing(frame_data, mode)
    
    mapping = {1: 'Interacting', 2: 'Not Interacting'} if mode == "binary" else {1: 'Interacting', 2: 'Available', 3: 'Alone'}
    frame_data['interaction_type'] = frame_data['interaction_type'].map(mapping)

    all_segments = []
    for video_id, video_df in frame_data.groupby('video_id'):
        all_segments.extend(create_segments_for_video(video_id, video_df))
    
    if not all_segments: return
    segments_df = pd.DataFrame(all_segments).sort_values(['video_id', 'start_time_sec']).reset_index(drop=True)

    segments_df = merge_same_segments(segments_df)
    default_fill = "Not Interacting" if mode == "binary" else "Alone"
    segments_df = fill_gaps_with_default(segments_df, default_type=default_fill)
    segments_df = merge_same_segments(segments_df)

    print_segment_summary(segments_df, mode)
    
    age_df = pd.read_csv(DataPaths.SUBJECTS_CSV_PATH, sep=";", decimal=",")[["video_name", "age_at_recording", "child_id"]]
    segments_df = segments_df.merge(age_df, on="video_name", how="left")
    
    segments_df.to_csv(output_file_path, index=False)
    print(f"✅ Saved results to {output_file_path}")

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['binary', 'tertiary'], default='tertiary')
    args = parser.parse_args()
    
    folder = Path(args.folder_path)
    input_path = list(folder.glob(f"{Inference.FRAME_LEVEL_INTERACTIONS_CSV.stem}*.csv"))[0]
    output_path = folder / Inference.INTERACTION_SEGMENTS_CSV.name

    main(output_path, input_path, mode=args.mode)
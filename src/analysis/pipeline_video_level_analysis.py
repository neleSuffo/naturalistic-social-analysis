import re
import argparse
import sys
import pandas as pd
import ruptures as rpt
from pathlib import Path

# Get the src directory (2 levels up from current notebook location)
src_path = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent
sys.path.append(str(src_path))

from constants import Analysis, DataPaths
from config import AnalysisConfig, DataConfig

# Constants
FPS = DataConfig.FPS # frames per second

def apply_cpd_smoothing(frame_data: pd.DataFrame, 
                        social_state_mode: str):
    """
    Applies Change Point Detection (CPD) smoothing to the frame-level data.
    Uses the PELT algorithm to identify regime changes based on presence scores.

    Parameters
    ----------
    frame_data : pd.DataFrame
        The input DataFrame containing frame-level multimodal flags and presence scores.
    social_state_mode : str
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
        breakpoints = algo.predict(pen=AnalysisConfig.CPD_PENALTY)

        start_idx = 0
        for end_idx in breakpoints:
            segment_slice = video_df.iloc[start_idx:end_idx]
            if not segment_slice.empty:
                counts = segment_slice['interaction_type'].value_counts(normalize=True)
                has_eng = (segment_slice['rule1_turn_taking'].any() | 
                           segment_slice['rule2_close_proximity'].any() | 
                           segment_slice['rule3_kcds_speaking'].any())
                
                if has_eng and counts.get(1, 0) >= AnalysisConfig.CPD_INTERACTING_THRESHOLD_LOW:
                    state = 1  # Interacting (Rule-supported)
                elif counts.get(1, 0) >= AnalysisConfig.CPD_INTERACTING_THRESHOLD:
                    state = 1  # Interacting (Density-supported)
                elif (counts.get(1, 0) + counts.get(2, 0)) >= AnalysisConfig.CPD_TOTAL_PRESENCE_FLOOR:
                    state = 2  # Available (or "Not Interacting" in binary)
                else:
                    # This is the "Alone" path. 
                    # In Binary mode, there is no state 3, so it MUST be state 2.
                    state = 3 if social_state_mode == "tertiary" else 2
    
                video_df.iloc[start_idx:end_idx, video_df.columns.get_loc('interaction_type')] = state
            start_idx = end_idx
        smoothed_results.append(video_df)
        
    return pd.concat(smoothed_results)

def create_segments_for_video(video_id: int, 
                              video_df: pd.DataFrame) -> list:
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
    
    # Initial segment
    curr_state = states[0]
    start_fr = frame_nums[0]
    
    # Iterate through frames to identify segment boundaries based on state changes
    for i in range(1, len(states)):
        if states[i] != curr_state:
            end_fr = frame_nums[i-1]
            dur = (end_fr - start_fr) / FPS
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
    segments.append({
        'video_id': video_id, 'video_name': video_name,
        'interaction_type': curr_state, 'segment_start': start_fr,
        'segment_end': end_fr, 'start_time_sec': start_fr / FPS, 
        'end_time_sec': end_fr / FPS, 'duration_sec': dur
    })
    return segments

def merge_same_segments(segments_df: pd.DataFrame):
    """
    Merge segments of the same category that have small gaps between them (less than or equal to AnalysisConfig.SAME_SEGMENT_MERGE_THRESHOLD seconds).
    ----------
    segments_df : pd.DataFrame
        DataFrame with segments.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with merged segments.
    """
    # Sort segments by video and start time
    merged_segments = []
    for _, video_segments in segments_df.groupby('video_id'):
        video_segments = video_segments.sort_values('start_time_sec').reset_index(drop=True)
        if len(video_segments) == 0: continue
        
        current_segment = video_segments.iloc[0].copy()
        for i in range(1, len(video_segments)):
            next_segment = video_segments.iloc[i]
            gap_duration = next_segment['start_time_sec'] - current_segment['end_time_sec']
            
            # Merge if same type AND gap is below threshold
            if (current_segment['interaction_type'] == next_segment['interaction_type'] and 
                gap_duration <= AnalysisConfig.SAME_SEGMENT_MERGE_THRESHOLD):
                current_segment['segment_end'] = next_segment['segment_end']
                current_segment['end_time_sec'] = next_segment['end_time_sec']
                current_segment['duration_sec'] = current_segment['end_time_sec'] - current_segment['start_time_sec']
            else:
                merged_segments.append(current_segment.to_dict())
                current_segment = next_segment.copy()
        merged_segments.append(current_segment.to_dict())
        
    return pd.DataFrame(merged_segments) if merged_segments else segments_df

def fill_gaps_with_default(segments_df):
    """
    Fills timeline gaps by either stretching segments or inserting default labels.
    If gap between segments is less than or equal to AnalysisConfig.GAP_STRETCH_THRESHOLD seconds, it stretches the previous segment to fill the gap.
    If the gap is larger, it inserts a new segment with a default label (e.g., "Not Interacting" or "Alone") for the duration of the gap.
    
    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame with interaction segments.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with a continuous timeline.
    """
    # Get default label based on mode
    default_type = AnalysisConfig.GAP_DEFAULT_LABEL_BINARY if "Not Interacting" in AnalysisConfig.GAP_DEFAULT_LABEL_BINARY else AnalysisConfig.GAP_DEFAULT_LABEL_TERTIARY
    
    filled_segments = []
    for video_id, video_df in segments_df.groupby('video_id'):
        v_segs = video_df.sort_values('start_time_sec').to_dict('records')
        for i in range(len(v_segs)):
            filled_segments.append(v_segs[i])
            if i < len(v_segs) - 1:
                gap = v_segs[i+1]['start_time_sec'] - v_segs[i]['end_time_sec']
                if 0 < gap <= AnalysisConfig.GAP_STRETCH_THRESHOLD:
                    filled_segments[-1]['end_time_sec'] = v_segs[i+1]['start_time_sec']
                    filled_segments[-1]['segment_end'] = v_segs[i+1]['segment_start'] - 1
                elif gap > AnalysisConfig.GAP_STRETCH_THRESHOLD:
                    filled_segments.append({
                        'video_id': video_id, 'video_name': v_segs[i]['video_name'],
                        'interaction_type': default_type, 'start_time_sec': v_segs[i]['end_time_sec'],
                        'end_time_sec': v_segs[i+1]['start_time_sec'], 
                        'segment_start': v_segs[i]['segment_end'] + 1,
                        'segment_end': v_segs[i+1]['segment_start'] - 1
                    })
    return pd.DataFrame(filled_segments)

def print_segment_summary(segments_df: pd.DataFrame, 
                          social_state_mode: str):
    """
    Print detailed summary statistics (minutes and percentages) for segments.
    
    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame with segments.
    social_state_mode : str
        "binary" or "tertiary".
    """
    if len(segments_df) > 0:
        total_min = round(segments_df['duration_sec'].sum() / 60, 2)
        print(f"\n📊 Final segment summary: {total_min} minutes total.")
        target_classes = ['Interacting', 'Not Interacting'] if social_state_mode == "binary" else ['Interacting', 'Alone', 'Available']
        for itype in target_classes:
            df_sub = segments_df[segments_df['interaction_type'] == itype]
            mins = round(df_sub['duration_sec'].sum() / 60, 2)
            perc = (mins / total_min * 100) if total_min > 0 else 0
            print(f"   {itype}: {len(df_sub)} segments ({mins}m - {perc:.1f}%)")
    else:
        print("\n📊 No segments created")

def main(output_file_path: Path, 
         frame_data_path: Path, 
         social_state_mode: str,
         hyperparameter_tuning: bool = False):
    """
    Main entry point for segment analysis. Loads data, smooths, segments, and saves.
    
    Parameters
    ----------
    output_file_path : Path
        Path to save the final segments CSV.
    frame_data_path : Path
        Path to the intermediate frame-level data CSV.
    social_state_mode : str
        "binary" or "tertiary" classification mode for social state analysis.
    hyperparameter_tuning: bool
        If True, takes configurations from the parent tuning script and avoids overwriting tuned parameters. Default is False.
    """
    # 2. Add the check
    if not hyperparameter_tuning:
        AnalysisConfig.apply_mode(social_state_mode)
    
    # Load frame-level data
    frame_data = pd.read_csv(frame_data_path)
    frame_data = apply_cpd_smoothing(frame_data, social_state_mode)
    
    # Map numeric states to labels based on mode
    mapping = AnalysisConfig.SOCIAL_STATE_MAPPING_BINARY if social_state_mode == "binary" else AnalysisConfig.SOCIAL_STATE_MAPPING_TERTIARY
    frame_data['interaction_type'] = frame_data['interaction_type'].map(mapping)

    # Create segments for each video and apply post-processing (merging, gap-filling)
    all_segments = []
    for video_id, video_df in frame_data.groupby('video_id'):
        all_segments.extend(create_segments_for_video(video_id, video_df))
    
    if not all_segments: return
    segments_df = pd.DataFrame(all_segments).sort_values(['video_id', 'start_time_sec']).reset_index(drop=True)

    # Post-processing: Merge same-type segments with small gaps and fill large gaps with default labels
    segments_df = merge_same_segments(segments_df)
    segments_df = fill_gaps_with_default(segments_df)
    segments_df = merge_same_segments(segments_df)

    print_segment_summary(segments_df, social_state_mode)
    
    age_df = pd.read_csv(DataPaths.SUBJECTS_CSV_PATH, sep=";", decimal=",")[["video_name", "age_at_recording", "child_id"]]
    segments_df = segments_df.merge(age_df, on="video_name", how="left")
    
    segments_df.to_csv(output_file_path, index=False)
    print(f"✅ Saved results to {output_file_path}")

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder_path', type=str, required=True)
    parser.add_argument('--social_state_mode', type=str, choices=['binary', 'tertiary'], default='tertiary')
    parser.add_argument('--hyperparameter_tuning', action='store_true')
    args = parser.parse_args()
    
    output_folder = Path(args.output_folder_path)
    frame_data_path = output_folder / Analysis.FRAME_LEVEL_INTERACTIONS_CSV.name
    output_file_path = output_folder / Analysis.INTERACTION_SEGMENTS_CSV.name

    main(output_file_path, 
         frame_data_path, 
         social_state_mode=args.social_state_mode, 
         hyperparameter_tuning=args.hyperparameter_tuning)
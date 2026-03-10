import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from constants import DataPaths, Inference, AudioClassification
from config import InferenceConfig

def parse_rttm(file_path: Path = AudioClassification.VTC_RTTM_FILE) -> pd.DataFrame:
    """Parses RTTM file into a DataFrame."""
    data = []
    if not file_path.exists():
        return pd.DataFrame()
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                data.append({
                    'video_name': parts[1],
                    'start_time_seconds': float(parts[3]),
                    'duration': float(parts[4]),
                    'end_time_seconds': float(parts[3]) + float(parts[4]),
                    'speaker': parts[7]
                })
    return pd.DataFrame(data)

def process_block(block):
    """
    Categorizes the block into one of four types:
    1. Successful Initiation (Child starts, turn occurs)
    2. Successful Response (Adult starts, turn occurs)
    3. Unanswered Child Bid (Child starts, no turn)
    4. Unanswered Adult Prompt (Adult starts, no turn)
    
    A block is defined as a sequence of vocalizations from the same speaker with gaps less than the defined thresholds.
    """
    if not block:
        return 0, 0, 0, 0, 0, 0
    
    # Calculate duration: End of last vocalization - Start of first vocalization
    block_duration = block[-1]['end_time_seconds'] - block[0]['start_time_seconds']
    
    turns = 0
    for i in range(1, len(block)):
        if block[i]['speaker'] != block[i-1]['speaker']:
            turns += 1
            
    # Initialize all counters
    succ_init, succ_resp, fail_init, fail_resp  = 0, 0, 0, 0
    first_speaker = block[0]['speaker']
    
    if turns > 0:
        if first_speaker == 'KCHI':
            succ_init = 1  # Successful exchange led by child
        else:
            succ_resp = 1  # Successful exchange led by adult
    else:
        if first_speaker == 'KCHI':
            fail_init = 1  # Child spoke, but no one replied
        else:
            fail_resp = 1  # Adult spoke, but child didn't reply
            
    return turns, succ_init, succ_resp, fail_init, fail_resp, block_duration

def count_directional_turns(vocalizations, segments_df) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Counts directional turn-taking within each interacting segment and measures individual child turn durations.
    
    Parameters:
    ----------
    vocalizations: DataFrame
        containing vocalization data with columns:
        - video_name
        - start_time_seconds
        - end_time_seconds
        - speaker (KCHI or KCDS)
    segments_df: DataFrame
        containing interaction segment data with columns:
        - child_id
        - video_name
        - age_at_recording
        - start_time_sec
        - end_time_sec
        - duration_sec
        - interaction_type
    
    Returns:
    --------
    results_df: DataFrame
        containing turn-taking counts and proportions for each segment
    raw_turns_df: DataFrame
        containing individual child turn durations and metadata for further analysis
    """
    results = []
    # list to store individual child turns
    raw_turns_df = []
    block_durations_raw = []
    MAX_TURN_GAP = InferenceConfig.MAX_TURN_TAKING_GAP_SEC
    MAX_SAME_GAP = InferenceConfig.MAX_SAME_SPEAKER_GAP_SEC

    for _, seg in segments_df.iterrows():
        # Only analyze 'Interacting' segments to maintain focused context
        if seg['interaction_type'] != 'Interacting':
            continue
            
        # Overlap Filter to capture all relevant speech within segment bounds
        seg_vocs = vocalizations[
            (vocalizations['video_name'] == seg['video_name']) & 
            (vocalizations['start_time_seconds'] < seg['end_time_sec']) & 
            (vocalizations['end_time_seconds'] > seg['start_time_sec'])
        ].copy().sort_values('start_time_seconds').reset_index(drop=True)
        
        # Filter for the Dyad (KCHI and KCDS)
        seg_vocs = seg_vocs[seg_vocs['speaker'].isin(['KCHI', 'KCDS'])].reset_index(drop=True)

        # Capture Individual Turn Durations
        child_only = seg_vocs[seg_vocs['speaker'] == 'KCHI']
        for _, voc in child_only.iterrows():
            raw_turns_df.append({
                'child_id': seg['child_id'],
                'age_at_recording': seg['age_at_recording'],
                'video_name': seg['video_name'],
                'vocalization_duration_sec': voc['duration'],
                'start_time': voc['start_time_seconds']
            })
            
        total_turns = 0
        s_init, s_resp, f_init, f_resp = 0, 0, 0, 0
        total_block_duration = 0
        
        if not seg_vocs.empty:
            current_block = [seg_vocs.iloc[0]]
            for i in range(1, len(seg_vocs)):
                prev = seg_vocs.iloc[i-1]
                curr = seg_vocs.iloc[i]
                gap = curr['start_time_seconds'] - prev['end_time_seconds']
                
                threshold = MAX_SAME_GAP if curr['speaker'] == prev['speaker'] else MAX_TURN_GAP
                
                if gap <= threshold:
                    current_block.append(curr)
                else:
                    t, si, sr, fi, fr, bd = process_block(current_block)
                    total_turns += t
                    s_init += si; s_resp += sr; f_init += fi; f_resp += fr
                    total_block_duration += bd
                    current_block = [curr]
            
            # Process final block
            t, si, sr, fi, fr, bd = process_block(current_block)
            total_turns += t
            s_init += si; s_resp += sr; f_init += fi; f_resp += fr
            total_block_duration += bd

        results.append({
            'child_id': seg['child_id'],
            'block_duration_sec': total_block_duration,
            'video_name': seg['video_name'],
            'age_at_recording': seg['age_at_recording'],
            'segment_start': seg['start_time_sec'],
            'segment_end': seg['end_time_sec'],
            'duration_sec': seg['duration_sec'],
            'total_turns': total_turns,
            'successful_initiations': s_init,
            'successful_responses': s_resp,
            'unanswered_child_bids': f_init,
            'unanswered_adult_prompts': f_resp,
            'segment_duration_minutes': seg['duration_sec'] / 60
        })
        
    return pd.DataFrame(results), pd.DataFrame(raw_turns_df)

def main():
    print("🗣️ RESEARCH QUESTION 4: TURN-TAKING ANALYSIS")
    print("=" * 70)
    
    # 1. Load Segments and Vocalizations
    segments_df = pd.read_csv(Inference.INTERACTION_SEGMENTS_CSV)
    all_vocalizations = parse_rttm()
    
    # 2. Categorize Social Blocks
    final_df, raw_turns_df = count_directional_turns(all_vocalizations, segments_df)
    
    # 3. Calculate Global Totals and Proportions
    final_df['total_attempts'] = (
        final_df['successful_initiations'] + final_df['successful_responses'] +
        final_df['unanswered_child_bids'] + final_df['unanswered_adult_prompts']
    )
    
    # Calculate percentages of the total social landscape
    for col in ['successful_initiations', 'successful_responses', 'unanswered_child_bids', 'unanswered_adult_prompts']:
        final_df[f'pct_{col}'] = (final_df[col] / final_df['total_attempts']).fillna(0)
    
    # Standard volume metric
    final_df['turns_per_minute'] = (final_df['total_turns'] / final_df['segment_duration_minutes']).fillna(0)
    
    # 4. Metadata Cleanup and Sort
    final_df['age_at_recording'] = (
        final_df['age_at_recording'].astype(str).str.replace('"', '').str.replace(',', '.').str.strip()
    )
    final_output = final_df.sort_values(['video_name', 'segment_start'])
    
    # 5. Save Results
    final_output.to_csv(Inference.TURN_TAKING_CSV, index=False)
    print(f"✅ Full four-category analysis saved to {Inference.TURN_TAKING_CSV}")
    
    # ----- Part 2: Child-Level Aggregation (Relative to Total Recording) -----
    # 1.G TRUE total recording duration for every child from source segments
    # (This includes Alone, Available, and Interacting time)
    child_total_durations = segments_df.groupby('child_id')['duration_sec'].sum().reset_index()
    child_total_durations.rename(columns={'duration_sec': 'total_recording_duration_sec'}, inplace=True)

    # 2. Get total turns from processed interacting segments
    child_turns_only = final_df.groupby('child_id')['total_turns'].sum().reset_index()

    # 3. Merge them to calculate the global rate
    child_level_turns = pd.merge(child_turns_only, child_total_durations, on='child_id', how='right').fillna(0)

    # 4. Add back age metadata (taking the minimum age found in the original segments)
    child_ages = segments_df.groupby('child_id')['age_at_recording'].min().reset_index()
    child_level_turns = pd.merge(child_level_turns, child_ages, on='child_id')

    # 5. Calculate global density (turns per minute of OVERALL recording time)
    child_level_turns['total_recording_minutes'] = child_level_turns['total_recording_duration_sec'] / 60
    child_level_turns['turns_per_minute'] = (
        child_level_turns['total_turns'] / child_level_turns['total_recording_minutes']
    ).fillna(0)

    # Save Child-Level Results
    child_level_turns.to_csv(Inference.GLOBAL_TURN_TAKING_CSV, index=False)
    print(f"✅ Child-level turn-taking (relative to total duration) saved to {Inference.GLOBAL_TURN_TAKING_CSV}")
    
    # ----- Part 3: SAVE INDIVIDUAL TURN DURATIONS -----
    if not raw_turns_df.empty:
        # Cleanup age strings in the raw durations dataframe
        raw_turns_df['age_at_recording'] = (
            raw_turns_df['age_at_recording'].astype(str)
            .str.replace('"', '').str.replace(',', '.')
            .str.strip()
        )
        
        # Save to the granular output path
        raw_turns_df.to_csv(Inference.TURN_DURATION_CSV, index=False)
        print(f"✅ Individual child turn durations saved to {Inference.TURN_DURATION_CSV}")
    else:
        print("⚠️ No child vocalizations found to save for turn duration analysis.")

if __name__ == "__main__":
    main()
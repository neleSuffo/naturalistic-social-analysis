import pandas as pd
import numpy as np
from constants import Analysis
from utils import parse_rttm, merge_overlapping_intervals

def main():
    print("🗣️ RESEARCH QUESTION 2: CHILD LANGUAGE PRODUCTION ANALYSIS")
    print("=" * 70)
    
    # 1. Load segments file
    segments_df = pd.read_csv(Analysis.INTERACTION_SEGMENTS_CSV)
    
    # 2. Extract Key Child (KCHI) speech from RTTM file
    kchi_vocalizations = parse_rttm(target_speech_types=['KCHI'])

    if kchi_vocalizations.empty:
        print("⚠️ Warning: No KCHI vocalizations found in RTTM file.")
        
    final_rows = []
    
    # 3. Iterate through EVERY segment from the segments CSV
    for _, seg in segments_df.iterrows():
        # Get vocalizations for this segment
        group = kchi_vocalizations[
            (kchi_vocalizations['video_name'] == seg['video_name']) & 
            (kchi_vocalizations['start_time_seconds'] < seg['end_time_sec']) & 
            (kchi_vocalizations['end_time_seconds'] > seg['start_time_sec'])
        ].copy()
                        
        if not group.empty:
            # Calculate overlap with the segment boundaries
            group['clipped_start'] = np.maximum(group['start_time_seconds'], seg['start_time_sec'])
            group['clipped_end'] = np.minimum(group['end_time_seconds'], seg['end_time_sec'])
            
        if group.empty:
            total_speech_seconds = 0.0
        else:
            intervals = list(zip(group['clipped_start'], group['clipped_end']))
            
            # Merge overlapping/adjacent intervals (RTTM can have overlaps)
            _, total_speech_seconds = merge_overlapping_intervals(intervals)
            
        final_rows.append({
            'child_id': seg['child_id'],
            'video_name': seg['video_name'],
            'age_at_recording': seg['age_at_recording'],
            'interaction_type': seg['interaction_type'],
            'segment_start_time': seg['start_time_sec'],
            'segment_end_time': seg['end_time_sec'],
            'total_speech_seconds': total_speech_seconds,
            'total_segment_duration': seg['duration_sec'],
            'segment_duration_minutes': seg['duration_sec'] / 60
        })
    
    # 4. Final Aggregation and Derived Columns
    final_df = pd.DataFrame(final_rows)
    final_df['speech_activity_percent'] = (
        final_df['total_speech_seconds'] / final_df['total_segment_duration']
    ).fillna(0)
    final_df['kchi_speech_minutes'] = final_df['total_speech_seconds'] / 60
    final_df['segment_duration_minutes'] = final_df['total_segment_duration'] / 60
    
    # Age Cleaning
    final_df['age_at_recording'] = (
        final_df['age_at_recording']
        .astype(str).str.replace('"', '').str.replace(',', '.').str.strip()
    )
    final_df['age_at_recording'] = pd.to_numeric(final_df['age_at_recording'], errors='coerce')
    
    # Save segment-level results
    final_df.to_csv(Analysis.KCS_SUMMARY_CSV, index=False)
    print(f"✅ KCS state summary saved to: {Analysis.KCS_SUMMARY_CSV}")
    
    # ----- PART 2A: Child-Level Aggregation ------
    child_level_summary = final_df.groupby('child_id').agg({
        'total_speech_seconds': 'sum',
        'total_segment_duration': 'sum',
        'age_at_recording': 'min'
    }).reset_index()

    # Calculate child-level percentage
    child_level_summary['speech_activity_percent'] = (
        child_level_summary['total_speech_seconds'] / 
        child_level_summary['total_segment_duration']
    ).fillna(0)

    # Convert to minutes for easier reading
    child_level_summary['total_speech_minutes'] = child_level_summary['total_speech_seconds'] / 60
    child_level_summary['total_recording_minutes'] = child_level_summary['total_segment_duration'] / 60

    # Save child-level results
    child_level_summary.to_csv(Analysis.GLOBAL_KCS_SUMMARY_CSV, index=False)
    print(f"✅ Child-level summary saved to: {Analysis.GLOBAL_KCS_SUMMARY_CSV}")
    
if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from constants import Analysis
from utils import parse_rttm, merge_overlapping_intervals

def main():
    print("🗣️ RESEARCH QUESTION 03: SPEECH EXPOSURE ANALYSIS")
    print("="*70)
    
    # 1. Load segments file
    segments_df = pd.read_csv(Analysis.INTERACTION_SEGMENTS_CSV)
    
    # 2. Extract both KCDS and OHS vocalizations from RTTM file
    all_vocalizations = parse_rttm(target_speech_types=['KCDS', 'OHS'])

    if all_vocalizations.empty:
        print("⚠️ Warning: No OHS or KCDS vocalizations found in RTTM file.")
            
    # 3. For each segment, calculate total speech exposure for KCDS and OHS separately, as well as combined
    exposure_categories = ['TOTAL', 'KCDS_ONLY', 'OHS_ONLY']
    final_rows = []

    # 3. Iterate through EVERY segment from the segments CSV
    for _, seg in segments_df.iterrows():
        # Get vocalizations for this segment
        group = all_vocalizations[
            (all_vocalizations['video_name'] == seg['video_name']) & 
            (all_vocalizations['start_time_seconds'] < seg['end_time_sec']) & 
            (all_vocalizations['end_time_seconds'] > seg['start_time_sec'])
        ].copy()

        if not group.empty:
            # Calculate overlap with the segment boundaries
            group['clipped_start'] = np.maximum(group['start_time_seconds'], seg['start_time_sec'])
            group['clipped_end'] = np.minimum(group['end_time_seconds'], seg['end_time_sec'])
        
        for exp_type in exposure_categories:
            if group.empty:
                speech_seconds = 0.0
            else:
                if exp_type == 'TOTAL':
                    data = group
                elif exp_type == 'KCDS_ONLY':
                    data = group[group['speech_type'] == 'KCDS']
                else:
                    data = group[group['speech_type'] == 'OHS']
                
                if data.empty:
                    speech_seconds = 0.0
                else:
                    intervals = list(zip(data['clipped_start'], data['clipped_end']))
                    _, speech_seconds = merge_overlapping_intervals(intervals)

            final_rows.append({
                'child_id': seg['child_id'],
                'video_name': seg['video_name'],
                'age_at_recording': seg['age_at_recording'],
                'interaction_type': seg['interaction_type'],
                'segment_start_time': seg['start_time_sec'],
                'segment_end_time': seg['end_time_sec'],
                'exposure_type': exp_type,
                'total_speech_seconds': speech_seconds,
                'total_segment_duration': seg['duration_sec'],
                'segment_duration_minutes': seg['duration_sec'] / 60
            })

    final_df = pd.DataFrame(final_rows)
    final_df['exposure_percent'] = (final_df['total_speech_seconds'] / final_df['total_segment_duration']).fillna(0)
        
    # Final cleanup and sort
    final_df = final_df.sort_values(['video_name', 'segment_start_time', 'exposure_type'])
    final_df.to_csv(Analysis.CDS_SUMMARY_CSV, index=False)
    print(f"✅ Clean results saved to {Analysis.CDS_SUMMARY_CSV}")
    
    
    # ----- PART 3A: Child-Level Aggregation ------
    # We group by both child_id and exposure_type to see the breakdown per child
    child_exposure_summary = final_df.groupby(['child_id', 'exposure_type']).agg({
        'total_speech_seconds': 'sum',
        'total_segment_duration': 'sum',
        'age_at_recording': 'min'
    }).reset_index()

    # Calculate global percentage for that specific exposure type
    child_exposure_summary['exposure_percent'] = (
        child_exposure_summary['total_speech_seconds'] / 
        child_exposure_summary['total_segment_duration']
    ).fillna(0)

    # Convert to minutes for easier reporting
    child_exposure_summary['total_speech_minutes'] = child_exposure_summary['total_speech_seconds'] / 60
    child_exposure_summary['total_recording_minutes'] = child_exposure_summary['total_segment_duration'] / 60

    # Save child-level results
    child_exposure_summary.to_csv(Analysis.GLOBAL_CDS_SUMMARY_CSV, index=False)
    
    print(f"✅ Child-level exposure summary saved to: {Analysis.GLOBAL_CDS_SUMMARY_CSV}")

if __name__ == "__main__":
    main()
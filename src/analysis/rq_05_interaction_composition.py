import pandas as pd
import argparse
import numpy as np
from pathlib import Path
from constants import Inference, DataPaths

def add_interaction_columns(frames_df, segments_df, mode="tertiary"):
    """
    Optimized version using GroupBy and Interval lookups.
    
    Parameters:
    ----------
    frames_df: pd.DataFrame 
        DataFrame with columns ['video_name', 'frame_number', 'proximity', ...]
    segments_df: pd.DataFrame
        DataFrame with columns ['video_name', 'segment_start', 'segment_end', 'interaction_type']
    mode: str
        "binary" for Interacting vs Not Interacting, "tertiary" for Interacting vs Alone vs Available
    """
    # 1. Setup column names
    target_cols = ['is_interaction']
    type_to_col = {'Interacting': 'is_interaction'}
    
    if mode == "binary":
        target_cols.append('is_not_interaction') 
        type_to_col['Not_Interacting'] = 'is_not_interaction'
    else:
        target_cols.extend(['is_alone', 'is_available'])
        type_to_col.update({'Alone': 'is_alone', 'Available': 'is_available'})

    # Initialize columns at once with False (more memory efficient than one-by-one)
    for col in target_cols:
        frames_df[col] = False

    # 2. Process by video to reduce search space
    # Convert frames_df to a dictionary of dataframes for O(1) access
    video_groups = dict(list(frames_df.groupby('video_name')))
    processed_fragments = []

    for video_name, segments in segments_df.groupby('video_name'):
        if video_name not in video_groups:
            continue
        
        v_frames = video_groups[video_name].copy()
        
        # For each segment in this specific video
        for _, seg in segments.iterrows():
            col = type_to_col.get(seg['interaction_type'])
            if col:
                # Use boolean indexing to set the column for the relevant frames
                mask = (v_frames['frame_number'] >= seg['segment_start']) & \
                       (v_frames['frame_number'] <= seg['segment_end'])
                v_frames.loc[mask, col] = True
        
        processed_fragments.append(v_frames)
        # Remove from dict to track which videos had segments
        del video_groups[video_name]

    # 3. Reconstruct the dataframe
    # Add back videos that had no segments (they stay all False)
    remaining_frames = list(video_groups.values())
    final_df = pd.concat(processed_fragments + remaining_frames).sort_index()
    
    return final_df

def main(mode="tertiary"):
    print(f"🚀 OPTIMIZED INTERACTION PROCESSING (Mode: {mode.upper()})")
    print("=" * 70)

    # Step 1: Load data (Keep your existing loading logic)
    try:
        segments_df = pd.read_csv(Inference.INTERACTION_SEGMENTS_CSV)
        frame_df_path = next(Path(Inference.FINAL_OUTPUT_FOLDER).glob("frame_level_social_interactions_*.csv"), None)
        
        if frame_df_path is None:
            raise FileNotFoundError("Frame-level interactions file not found.")
            
        frames_df = pd.read_csv(frame_df_path)
        age_df = pd.read_csv(DataPaths.SUBJECTS_CSV_PATH, sep=';')
    except (FileNotFoundError, StopIteration) as e:
        print(f"❌ Error loading data: {e}")
        return

    # Step 2: Optimized Processing
    frames_df = add_interaction_columns(frames_df, segments_df, mode=mode)
    
    # Step 3: Fast Metadata Merge
    # Pre-clean age_df to avoid doing it inside the main dataframe later
    age_df = age_df[['video_name', 'age_at_recording', 'child_id']].copy()
    age_df['age_at_recording'] = (
        age_df['age_at_recording']
        .astype(str)
        .str.replace('"', '', regex=False)
        .str.replace(',', '.', regex=False)
        .str.strip()
    )
    age_df['age_at_recording'] = pd.to_numeric(age_df['age_at_recording'], errors='coerce')

    # Vectorized operations
    frames_df['proximity_filled'] = frames_df['proximity'].fillna(-1)
    frames_df = frames_df.merge(age_df, on='video_name', how='left')

    # Step 4: Save (Using compression if file is huge can be slower but saves space)
    output_path = Inference.INTERACTION_COMPOSITION_CSV
    frames_df.to_csv(output_path, index=False)

    # adjust print message based on mode
    if mode == "binary":
        print(f"\n✅ Done! Interacting: {frames_df['is_interaction'].sum()} frames, Not Interacting: {frames_df['is_not_interaction'].sum()} frames.")
    else:         
        print(f"\n✅ Done! Interacting: {frames_df['is_interaction'].sum()} frames, Alone: {frames_df['is_alone'].sum()} frames, Available: {frames_df['is_available'].sum()} frames.")
    print(f"📄 Saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['binary', 'tertiary'], default='tertiary')
    args = parser.parse_args()
        
    main(mode=args.mode)
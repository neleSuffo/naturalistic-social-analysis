import pandas as pd
import argparse
from pathlib import Path
from constants import Analysis

def add_interaction_columns(frames_df: pd.DataFrame, 
                            segments_df: pd.DataFrame, 
                            social_state_mode="tertiary") -> pd.DataFrame:
    """
    Adds binary columns to frames_df indicating whether each frame falls within an interaction segment, and if so, what type of interaction.
    
    Parameters:
    ----------
    frames_df: pd.DataFrame 
        DataFrame with columns ['video_name', 'frame_number', 'proximity', ...]
    segments_df: pd.DataFrame
        DataFrame with columns ['video_name', 'segment_start', 'segment_end', 'interaction_type']
    social_state_mode: str
        "binary" for Interacting vs Not Interacting, "tertiary" for Interacting vs Alone vs Available
    """
    # 1. Setup column names
    target_cols = ['is_interaction']
    type_to_col = {'Interacting': 'is_interaction'}
    
    if social_state_mode == "binary":
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

def main(social_state_mode: str = 'tertiary',
         output_folder: Path = None):
    """
    Generate frame-level interaction composition by merging segment-level interactions with frame-level metadata.

    Parameters
    ----------
    social_state_mode : str, optional
        Whether to use "binary" (Interacting vs Not Interacting) or "tertiary" (Interacting vs Alone vs Available) classification, by default "tertiary"
    output_folder : Path, optional
        Optional output folder path to save the interaction composition CSV. If not provided, saves to default location defined in constants, by default None
        
    Raises
    ------
    FileNotFoundError
        _description_
    """
    print(f"🚀 INTERACTION PROCESSING (Mode: {social_state_mode.upper()})")
    print("=" * 70)

    # Step 1: Load data
    if output_folder:
        segments_path = output_folder / Analysis.INTERACTION_SEGMENTS_CSV.name
        frames_path = output_folder / Analysis.FRAME_LEVEL_INTERACTIONS_CSV.name
    else:
        segments_path = Analysis.INTERACTION_SEGMENTS_CSV
        frames_path = Analysis.FRAME_LEVEL_INTERACTIONS_CSV
    segments_df = pd.read_csv(segments_path)
    frames_df = pd.read_csv(frames_path)
    
    # Step 2: Optimized Processing
    frames_df = add_interaction_columns(frames_df, segments_df, social_state_mode=social_state_mode)
    
    # # 3. Fast Metadata Merge
    print("📝 Syncing metadata from segments...")
    metadata_map = segments_df[['video_name', 'age_at_recording', 'child_id']].drop_duplicates()

    # Ensure age is numeric
    metadata_map['age_at_recording'] = pd.to_numeric(
        metadata_map['age_at_recording'].astype(str)
        .str.replace('"', '', regex=False)
        .str.replace(',', '.', regex=False)
        .str.strip(), 
        errors='coerce'
    )
    # Vectorized operations
    frames_df = frames_df.merge(metadata_map, on='video_name', how='left')
    frames_df['proximity_filled'] = frames_df['proximity'].fillna(-1)

    # Step 4: Save
    if output_folder:
        output_path = output_folder / Analysis.INTERACTION_COMPOSITION_CSV.name
    else:
        output_path = Analysis.INTERACTION_COMPOSITION_CSV
    frames_df.to_csv(output_path, index=False)

    # adjust print message based on social_state_mode
    if social_state_mode == "binary":
        print(f"\n✅ Done! Interacting: {frames_df['is_interaction'].sum()} frames, Not Interacting: {frames_df['is_not_interaction'].sum()} frames.")
    else:         
        print(f"\n✅ Done! Interacting: {frames_df['is_interaction'].sum()} frames, Alone: {frames_df['is_alone'].sum()} frames, Available: {frames_df['is_available'].sum()} frames.")
    print(f"📄 Saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--social_state_mode', type=str, choices=['binary', 'tertiary'], default='tertiary')
    parser.add_argument('--output_folder', type=str, default=None, help="Optional output folder path to save the interaction composition CSV")
    args = parser.parse_args()
        
    main(social_state_mode=args.social_state_mode, 
         output_folder=Path(args.output_folder) if args.output_folder else None)
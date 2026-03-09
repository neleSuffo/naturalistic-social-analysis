import re
import argparse
import sys
import pandas as pd
import shutil
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
    """
    type_map = {
        'Interacting': InferenceConfig.MIN_INTERACTING_SEGMENT_DURATION_SEC,
        'Alone': InferenceConfig.MIN_ALONE_SEGMENT_DURATION_SEC,
        'Available': InferenceConfig.MIN_AVAILABLE_SEGMENT_DURATION_SEC,
        'Not Interacting': InferenceConfig.MIN_NOT_INTERACTING_SEGMENT_DURATION_SEC,
    }
    # Use a default minimum (e.g., the largest) if the type is unexpected
    default_min = getattr(InferenceConfig, 'MIN_ALONE_SEGMENT_DURATION_SEC', 15.0) 
    return type_map.get(interaction_type, default_min)

def create_segments_for_video(video_id, video_df):
    """
    Create segments for a single video. Buffers short state changes and enforces minimum segment durations.
    
    Parameters
    ----------
    video_id : int
        Video identifier
    video_df : pd.DataFrame
        Frame-level data for this video
        
    Returns
    -------
    list
        List of segment dictionaries
    """
    video_df = video_df.sort_values('frame_number').reset_index(drop=True)
    
    if len(video_df) == 0:
        return []
                
    # Get interaction states and frame numbers
    states = video_df['interaction_type'].values
    frame_numbers = video_df['frame_number'].values
    video_name = video_df['video_name'].iloc[0]
    
    segments = []
    current_state = states[0]
    segment_start_frame = frame_numbers[0]
    
    # Process state changes
    for i in range(1, len(states)):
        if states[i] != current_state:
            segment_end_frame = frame_numbers[i-1]
            
            required_min_duration = get_min_segment_duration(current_state)
            
            segment_duration = (segment_end_frame - segment_start_frame) / FPS
            if segment_duration >= required_min_duration:
                
                segments.append({
                    'video_id': video_id,
                    'video_name': video_name,
                    'interaction_type': current_state,
                    'segment_start': segment_start_frame,
                    'segment_end': segment_end_frame,
                    'start_time_sec': segment_start_frame / FPS,
                    'end_time_sec': segment_end_frame / FPS,
                    'duration_sec': segment_duration
                })
            
            current_state = states[i]
            segment_start_frame = frame_numbers[i]
    
    # Handle the final segment
    segment_end_frame = frame_numbers[-1]
    
    required_min_duration = get_min_segment_duration(current_state)
    
    segment_duration = (segment_end_frame - segment_start_frame) / FPS
    if segment_duration >= required_min_duration:
        
        segments.append({
            'video_id': video_id,
            'video_name': video_name,
            'interaction_type': current_state,
            'segment_start': segment_start_frame,
            'segment_end': segment_end_frame,
            'start_time_sec': segment_start_frame / FPS,
            'end_time_sec': segment_end_frame / FPS,
            'duration_sec': segment_duration
        })
    
    return segments

def merge_same_segments(segments_df, max_gap_sec=100):
    """
    Merge segments of the same category that have small gaps between them.
    
    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame with segments
    max_gap_sec : float
        The maximum duration (seconds) allowed between segments of the same type to merge them.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with merged segments
    """
    merged_segments = []
    
    for video_id, video_segments in segments_df.groupby('video_id'):
        video_segments = video_segments.sort_values('start_time_sec').reset_index(drop=True)
        
        if len(video_segments) == 0:
            continue
        
        current_segment = video_segments.iloc[0].copy()
        
        for i in range(1, len(video_segments)):
            next_segment = video_segments.iloc[i]
            
            gap_duration = next_segment['start_time_sec'] - current_segment['end_time_sec']
            
            if (current_segment['interaction_type'] == next_segment['interaction_type'] and 
                gap_duration <= max_gap_sec):
                
                current_segment['segment_end'] = next_segment['segment_end']
                current_segment['end_time_sec'] = next_segment['end_time_sec']
                current_segment['duration_sec'] = (
                    current_segment['end_time_sec'] - current_segment['start_time_sec']
                )
                
            else:
                merged_segments.append(current_segment.to_dict())
                current_segment = next_segment.copy()
        
        merged_segments.append(current_segment.to_dict())
    
    return pd.DataFrame(merged_segments) if merged_segments else segments_df

def reclassify_implicit_turn_taking(segments_df, frames_by_video, mode="tertiary"):
    """
    Reclassify 'Available' or 'Alone' segments to 'Interacting' if they contain 
    sufficient evidence of implicit, KCHI-based turn-taking.

    Criteria (Implicit Turn-Taking Character):
    1. Segment type is 'Available' or 'Alone'.
    2. At least 20% of the segment's sampled frames are KCHI-only (KCHI=1, CDS=0, OHS=0).
    3. Person/Face (person_present) is detected for at least 5% of the segment's frames.
    
    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame with segments (after buffering and merging).
    frames_by_video : dict
        Dictionary mapping video_name to its frame-level DataFrame.
    mode : str
        "binary" or "tertiary" to determine target segments for reclassification.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with reclassified segments.
    """    
    updated_segments_df = segments_df.copy()
    reclassified_count = 0

    for index, segment in segments_df.iterrows():
        target_types = ['Not Interacting'] if mode == "binary" else ['Available', 'Alone']
        if segment['interaction_type'] not in target_types:
            continue
        
        v_frames = frames_by_video.get(segment['video_name'])
        if v_frames is None: continue
            
        seg_frames = v_frames[
            (v_frames['frame_number'] >= segment['segment_start']) & 
            (v_frames['frame_number'] <= segment['segment_end'])
        ]
        
        total_segment_frames = len(seg_frames)
        if total_segment_frames == 0: continue
            
        kchi_only_count = ((seg_frames['has_kchi'] == 1) & (seg_frames['has_cds'] == 0)).sum()
        kchi_only_fraction = kchi_only_count / total_segment_frames

        if kchi_only_fraction < InferenceConfig.KCHI_ONLY_FRACTION_THRESHOLD:
            continue
            
        person_presence_count = (seg_frames['person_or_face_present'] == 1).sum()
        person_presence_fraction = person_presence_count / total_segment_frames
        
        if person_presence_fraction >= InferenceConfig.MIN_PERSON_PRESENCE_FRACTION:
            updated_segments_df.at[index, 'interaction_type'] = 'Interacting'
            reclassified_count += 1
            
    print(f"   Reclassified {reclassified_count} segments to 'Interacting' (Implicit Turn-Taking).")
    return updated_segments_df

def reclassify_alone_segments(segments_df, frames_by_video, detection_col='person_or_face_present'):
    """
    Reclassify 'Alone' segments to 'Available' if they contain sufficient evidence
    of partner presence (visual or audio) that exceeds defined thresholds.
    
    (EXCLUDES segments where the frame-level classification was ALONE due to MEDIA.)
    
    CRITERIA FOR RECLASSIFICATION (Alone -> Available):
    1. Segment type must be 'Alone'.
    2. Segment length must be longer than MIN_RECLASSIFY_DURATION_SEC (to avoid short noise).
    3. Person/Face detection (person_or_face_present) must occur for > 5% of segment frames.
    4. OR Partner audio (has_cds OR has_ohs) must occur for > 5% of segment frames.
    
    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame with final interaction segments.
    frames_by_video : dict
        Dictionary mapping video_name to its frame-level DataFrame.
    detection_col : str
        The name of the fused detection column.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with reclassified segments.
    """        
    updated_segments_df = segments_df.copy()
    reclassified_count = 0

    for index, segment in updated_segments_df.iterrows():
        if segment['interaction_type'] != 'Alone': continue
        if segment['duration_sec'] <= InferenceConfig.MIN_RECLASSIFY_DURATION_SEC: continue

        v_frames = frames_by_video.get(segment['video_name'])
        if v_frames is None: continue
        
        seg_frames = v_frames[
            (v_frames['frame_number'] >= segment['segment_start']) & 
            (v_frames['frame_number'] <= segment['segment_end'])
        ]

        if seg_frames.empty or seg_frames['is_media_interaction'].any(): continue

        visual_frac = (seg_frames[detection_col] == 1).sum() / len(seg_frames)
        audio_frac = ((seg_frames['has_cds'] == 1) | (seg_frames['has_ohs'] == 1)).sum() / len(seg_frames)

        if visual_frac > InferenceConfig.ALONE_RECLASSIFY_VISUAL_THRESHOLD or \
           audio_frac > InferenceConfig.ALONE_RECLASSIFY_AUDIO_THRESHOLD:
            updated_segments_df.at[index, 'interaction_type'] = 'Available'
            reclassified_count += 1
            
    print(f"   Reclassified {reclassified_count} 'Alone' segments to 'Available'.")
    return updated_segments_df

def reclassify_ghost_segments(segments_df, frames_by_video):
    """
    Reclassifies 'Interacting' or 'Available' segments to 'Alone' if they 
    lack sufficient visual human presence, using type-specific thresholds.
    
    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame with interaction segments.
    frames_by_video : dict
        Dictionary mapping video_name to its frame-level DataFrame.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with reclassified segments.
    """
    updated_segments_df = segments_df.copy()
    reclassified_count = 0

    for index, segment in segments_df.iterrows():
        if segment['interaction_type'] == 'Alone': continue
        
        if segment['interaction_type'] == 'Interacting':
            min_duration = InferenceConfig.MIN_GHOST_CHECK_DURATION_INTERACTING
            visual_threshold = InferenceConfig.GHOST_VISUAL_THRESHOLD_INTERACTING
        else:
            min_duration = InferenceConfig.MIN_GHOST_CHECK_DURATION_AVAILABLE
            visual_threshold = InferenceConfig.GHOST_VISUAL_THRESHOLD_AVAILABLE

        if segment['duration_sec'] < min_duration: continue
        
        v_frames = frames_by_video.get(segment['video_name'])
        if v_frames is None: continue

        seg_frames = v_frames[
            (v_frames['frame_number'] >= segment['segment_start']) & 
            (v_frames['frame_number'] <= segment['segment_end'])
        ]
        
        if not seg_frames.empty:
            human_presence_frac = (seg_frames['person_or_face_present'] == 1).sum() / len(seg_frames)
            if human_presence_frac < visual_threshold:
                updated_segments_df.at[index, 'interaction_type'] = 'Alone'
                reclassified_count += 1

    print(f"   Reclassified {reclassified_count} 'Ghost' segments to 'Alone'.")
    return updated_segments_df

def fill_gaps_with_default(segments_df, default_type="Alone"):
    """
    Fills gaps in the timeline by either stretching existing segments (small gaps) 
    or inserting default interaction types (large gaps).
    
    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame with interaction segments.
    default_type : str
        The interaction type to use for filling large gaps.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with filled gaps.
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
                        'video_id': video_id,
                        'video_name': v_segs[i]['video_name'],
                        'interaction_type': default_type,
                        'start_time_sec': v_segs[i]['end_time_sec'],
                        'end_time_sec': v_segs[i+1]['start_time_sec'],
                        'segment_start': v_segs[i]['segment_end'] + 1,
                        'segment_end': v_segs[i+1]['segment_start'] - 1
                    })
    return pd.DataFrame(filled_segments)

def print_segment_summary(segments_df, mode):
    """
    Print summary statistics for the created segments.
    
    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame with segments
    mode : str
        "binary" or "tertiary" to determine which interaction types to summarize
    """
    if len(segments_df) > 0:
        total_segments = len(segments_df)
        # Recalculate duration for all segments after reclassification
        segments_df['duration_sec'] = segments_df['end_time_sec'] - segments_df['start_time_sec']

        # Calculate total duration in minutes
        total_duration = round(segments_df['duration_sec'].sum() / 60, 2)
        
        print(f"\n📊 Final segment summary:")
        print(f"   Total segments: {total_segments} ({total_duration} minutes)")

        if mode == "binary":
            for itype in ['Interacting', 'Not Interacting']:
                df_sub = segments_df[segments_df['interaction_type'] == itype]
                count = len(df_sub)
                minutes = round(df_sub['duration_sec'].sum() / 60, 2)
                percent = (minutes / total_duration * 100) if total_duration > 0 else 0
                print(f"   {itype}: {count} ({minutes} minutes - {percent:.1f}%)")
        else:
            for itype in ['Interacting', 'Alone', 'Available']:
                df_sub = segments_df[segments_df['interaction_type'] == itype]
                count = len(df_sub)
                minutes = round(df_sub['duration_sec'].sum() / 60, 2)
                percent = (minutes / total_duration * 100) if total_duration > 0 else 0
                print(f"   {itype}: {count} ({minutes} minutes - {percent:.1f}%)")
    else:
        print("\n📊 No segments created")

def main(output_file_path: Path, frame_data_path: Path, hyperparameter_tuning: bool = False, mode: str = "tertiary"):
    """
    Main entry point for video-level segment analysis.
    Processes frame-level interaction data through segment creation, merging, 
    reclassification, and gap filling.
    
    hyperparameter_tuning: Bool
        Whether script runs in hyperparmeter mode or not 
    """     
    # --- Hyperparameter Tuning Logic ---
    if hyperparameter_tuning:
        run_dir = output_file_path.parent
        try:
            script_path = Path(__file__)
            shutil.copy(script_path, run_dir / script_path.name)
        except NameError:
            print("⚠️ __file__ not defined, skipping script copy.")

    # Step 1: Load and Optimize
    frame_data = pd.read_csv(frame_data_path)
    frames_by_video = dict(list(frame_data.groupby('video_name')))

    all_segments = []
    for video_id, video_df in frame_data.groupby('video_id'):
        all_segments.extend(create_segments_for_video(video_id, video_df))
    
    if not all_segments:
        print("❌ No segments could be created."); return

    segments_df = pd.DataFrame(all_segments).sort_values(['video_id', 'start_time_sec']).reset_index(drop=True)

    segments_df = merge_same_segments(segments_df)
    
    if mode == "tertiary":
        segments_df = reclassify_alone_segments(segments_df, frames_by_video)
    
    segments_df = reclassify_implicit_turn_taking(segments_df, frames_by_video, mode=mode)

    print("🧹 Final consolidation...")
    segments_df = merge_same_segments(segments_df)
        
    if mode == "tertiary":
        segments_df = reclassify_ghost_segments(segments_df, frames_by_video)

    default_fill = "Not Interacting" if mode == "binary" else "Available"
    segments_df = fill_gaps_with_default(segments_df, default_type=default_fill)
    segments_df = merge_same_segments(segments_df)

    print_segment_summary(segments_df, mode)
    
    age_df = pd.read_csv(DataPaths.SUBJECTS_CSV_PATH, sep=";", decimal=",")[["video_name", "age_at_recording", "child_id"]]
    segments_df = segments_df.merge(age_df, on="video_name", how="left")
    
    segments_df.to_csv(output_file_path, index=False)
    print(f"✅ Final interaction segments saved to {output_file_path}")

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Video-level social interaction segment analysis')
    parser.add_argument('--folder_path', type=str, required=True, help='Folder path containing input CSV and where outputs will be saved')
    parser.add_argument('--mode', type=str, choices=['binary', 'tertiary'], default='tertiary',
                    help='Binary: Interacting vs Not Interacting. Tertiary: Interacting, Available, Alone.')    
    args = parser.parse_args()
    
    folder_path = Path(args.folder_path)
    input_path = list(folder_path.glob(f"{Inference.FRAME_LEVEL_INTERACTIONS_CSV.stem}*.csv"))[0]
    output_path = folder_path / Inference.INTERACTION_SEGMENTS_CSV.name
    prefix = Inference.FRAME_LEVEL_INTERACTIONS_CSV.stem
    matching_files = list(folder_path.glob(f"{prefix}*.csv"))

    if not matching_files:
        raise FileNotFoundError(
            f"No file starting with '{prefix}' found in {folder_path}"
        )
    elif len(matching_files) > 1:
        print(f"⚠️ Multiple files found starting with '{prefix}', using the first one:")
        for f in matching_files:
            print(f"   - {f.name}")

    input_path = matching_files[0]
    print(f"Using input frame-level data from: {input_path}")
    # Run main analysis
    main(output_file_path=output_path, frame_data_path=input_path, hyperparameter_tuning=False, mode=args.mode)

    # Copy current script into folder for reproducibility
    try:
        current_script = Path(__file__)
        destination_script = folder_path / current_script.name
        shutil.copy(current_script, destination_script)
        print(f"🧾 Copied script to {destination_script}")
    except Exception as e:
        print(f"⚠️ Could not copy script to folder: {e}")
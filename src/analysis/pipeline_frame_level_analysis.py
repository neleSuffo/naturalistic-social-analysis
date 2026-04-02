"""
Frame-Level Social Interaction Analysis Pipeline

This script integrates multimodal data (Face, Person, Audio, Books) from a SQLite 
database and classifies each frame into a social state (Interacting, Available, Alone).
Optimization: Uses vectorized Pandas operations instead of row-wise processing.
"""

import logging
import sqlite3
import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict

# Path configuration
src_path = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent
sys.path.append(str(src_path))

from constants import DataPaths, Analysis
from config import AnalysisConfig, DataConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_persistent_tables(conn):
    """
    Creates persistent indexed tables in SQLite to optimize repeated queries.
    Pre-aggregates detections to avoid re-calculating exclusions and MAX() values.
    
    Parameters
    ----------
    conn: sqlite3.Connection
        Active connection to the SQLite database.
        
    This function creates:
    1. PersistentExclusions: A table of detection IDs that should be excluded (e.g., faces inside books).
    2. CachedFaceAgg: A pre-aggregated table of maximum proximity and face confidence per frame (after exclusions).
    3. CachedPersonAgg: A pre-aggregated table of maximum person confidence per frame (after exclusions).
    Indexes are created on these tables to speed up JOIN operations in the main query.
    """
    sample_rate = AnalysisConfig.SAMPLE_RATE

    # 1. Persistent Exclusion Table (Detections inside Books)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS PersistentExclusions AS
    SELECT fd.detection_id, 'face' as type FROM FaceDetections fd
    JOIN BookDetections bd ON fd.frame_number = bd.frame_number AND fd.video_id = bd.video_id
    WHERE fd.x_min >= bd.x_min AND fd.y_min >= bd.y_min AND fd.x_max <= bd.x_max AND fd.y_max <= bd.y_max
    UNION ALL
    SELECT pc.detection_id, 'person' as type FROM PersonDetections pc
    JOIN BookDetections bd ON pc.frame_number = bd.frame_number AND pc.video_id = bd.video_id
    WHERE pc.x_min >= bd.x_min AND pc.y_min >= bd.y_min AND pc.x_max <= bd.x_max AND pc.y_max <= bd.y_max;
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pers_excl ON PersistentExclusions(detection_id);")

    # 2. Aggregated Face Table (Filtered and Sampled)
    # choose MAX() for proximity and confidence to capture the strongest signal per frame, after exclusions
    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS CachedFaceAgg AS
    SELECT frame_number, video_id, MAX(proximity) AS proximity, MAX(confidence_score) AS face_conf
    FROM FaceDetections
    WHERE detection_id NOT IN (SELECT detection_id FROM PersistentExclusions WHERE type='face')
      AND frame_number % {sample_rate} = 0
    GROUP BY video_id, frame_number;
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_face_agg ON CachedFaceAgg(video_id, frame_number);")

    # 3. Aggregated Person Table (Filtered and Sampled)
    # choose MAX() for confidence to capture the strongest signal per frame, after exclusions
    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS CachedPersonAgg AS
    SELECT frame_number, video_id, MAX(confidence_score) AS person_conf
    FROM PersonDetections
    WHERE detection_id NOT IN (SELECT detection_id FROM PersistentExclusions WHERE type='person')
      AND frame_number % {sample_rate} = 0
    GROUP BY video_id, frame_number;
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_person_agg ON CachedPersonAgg(video_id, frame_number);")

def get_all_analysis_data(conn, video_list: list) -> pd.DataFrame:
    """
    Fetches and integrates all necessary data for frame-level analysis in a single optimized query.
    
    Parameters
    ----------
    conn: sqlite3.Connection
        Active connection to the SQLite database.
    video_list: list
        List of video names to include in the analysis. If empty, includes all videos.
        
    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per sampled frame, containing:
        - frame_number
        - video_id
        - video_name
        - proximity (max proximity from face detections)
        - face_conf (max face confidence)
        - person_conf (max person confidence)
        - has_kchi (binary flag for presence of KCHI in audio)
        - has_ohs (binary flag for presence of OHS in audio)
        - has_cds (binary flag for presence of CDS in audio)
        - instant_presence_conf (max of face_conf and person_conf, used for gating)
    """
    sample_rate = AnalysisConfig.SAMPLE_RATE
    placeholders = ','.join('?' for _ in video_list)
    
    query = f"""
    WITH RECURSIVE FrameGrid AS (
        SELECT video_id, video_name, 0 AS frame_number, max_frame FROM Videos
        WHERE video_name IN ({placeholders})
        UNION ALL
        SELECT video_id, video_name, frame_number + {sample_rate}, max_frame FROM FrameGrid
        WHERE frame_number + {sample_rate} <= max_frame
    )
    SELECT
        fg.frame_number, fg.video_id, fg.video_name,
        COALESCE(fa.proximity, 0) AS proximity,
        COALESCE(fa.face_conf, 0) AS face_conf,
        COALESCE(pa.person_conf, 0) AS person_conf,
        COALESCE(af.has_kchi, 0) AS has_kchi,
        COALESCE(af.has_ohs, 0) AS has_ohs,
        COALESCE(af.has_cds, 0) AS has_cds,
        MAX(COALESCE(pa.person_conf, 0), COALESCE(fa.face_conf, 0)) AS instant_presence_conf
    FROM FrameGrid fg
    LEFT JOIN CachedFaceAgg fa ON fg.frame_number = fa.frame_number AND fg.video_id = fa.video_id
    LEFT JOIN CachedPersonAgg pa ON fg.frame_number = pa.frame_number AND fg.video_id = pa.video_id
    LEFT JOIN AudioClassifications af ON fg.frame_number = af.frame_number AND fg.video_id = af.video_id
    ORDER BY fg.video_id, fg.frame_number
    """
    return pd.read_sql(query, conn, params=tuple(video_list))

def calculate_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized calculation of rolling window features (Presence Score, Persistence). 
    In detail, this function computes:
    1. Presence Score: A rolling average of the instant presence confidence to smooth out short-term fluctuations.
    2. Visual Persistence: A binary feature indicating if a person has been seen recently, based on high-confidence anchors and short-term presence.
    3. Sustained Audio Windows: Binary features indicating if there has been sustained CDS or OHS presence over a defined window.   
    
    Parameters    
    ----------
    df: pd.DataFrame
        Input DataFrame containing frame-level data.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional windowed features.
    """
    fps = DataConfig.FPS
    sr = AnalysisConfig.SAMPLE_RATE
    samples_per_sec = fps / sr
    
    # Rolling presence signal for CPD 
    # calculate a smoothed presence score using a rolling average of the instant presence confidence
    df['presence_score'] = df['instant_presence_conf'].rolling(
        window=int(samples_per_sec), min_periods=1, center=True
    ).mean().fillna(0)

    # Visual Persistence
    df['is_high_conf_anchor'] = ((df['proximity'] > AnalysisConfig.HIGH_CONFIDENCE_PROXIMITY_THRESHOLD) | (df['face_conf'] > AnalysisConfig.HIGH_CONFIDENCE_FACE_CONFIDENCE)).astype(int)  
    
    # Use rolling max to determine if a person has been seen recently, either through a high-confidence anchor (long memory) or short-term presence (flicker)
    long_mem = df['is_high_conf_anchor'].rolling(
        window=int(AnalysisConfig.VISUAL_PERSISTENCE_SEC * samples_per_sec), 
        min_periods=1, center=True
    ).max().fillna(0)
    
    short_mem = df['instant_presence_conf'].rolling(
        window=int(AnalysisConfig.SHORT_TERM_VISUAL_MEMORY_SEC * samples_per_sec), min_periods=1, center=True
    ).max().fillna(0) >= AnalysisConfig.INSTANT_CONFIDENCE_THRESHOLD

    df['person_seen_recently'] = (long_mem == 1) | (short_mem)

    # Sustained Audio Windowing
    sustained_window = int(AnalysisConfig.SUSTAINED_KCDS_WINDOW_SEC * samples_per_sec)
    df['is_sustained_kcds'] = df['has_cds'].rolling(window=sustained_window).mean() >= AnalysisConfig.SUSTAINED_KCDS_THRESHOLD
    df['is_sustained_ohs'] = df['has_ohs'].rolling(window=sustained_window).mean() >= AnalysisConfig.MIN_PRESENCE_OHS_FRACTION

    return df

def classify_frames(df: pd.DataFrame, social_state_mode: str = "tertiary") -> pd.DataFrame:
    """
    Applies vectorized Boolean logic to classify each frame into social states based on the defined rules.
    The classification is done in a single pass using Pandas operations, following the priority of rules:
    1. Interacting (1): If any of the turn-taking, close proximity, or sustained KCDS rules are met.
    2. Available (2): If the person is seen recently (visual persistence) or if there is sustained OHS presence.
    3. Alone (3): Default state if neither of the above conditions are met.
    In "binary" mode, Available and Alone are merged into a single Not-Interacting class (2), while Interacting remains 1.
        
    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame with calculated features.
    social_state_mode: str
        "tertiary" for three-class classification, "binary" to merge Available and Alone into one class.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with an additional 'interaction_type' column indicating the classified social state for each frame.
    """
    # 1. Define Boolean Logic Filters
    is_visual_anchor = df['presence_score'] >= AnalysisConfig.AUDIO_VISUAL_GATING_FLOOR
    is_available_visual = df['person_seen_recently'].astype(bool)
    is_visual_confident = df['instant_presence_conf'] >= AnalysisConfig.INSTANT_CONFIDENCE_THRESHOLD
    is_close = df['proximity'] >= AnalysisConfig.PROXIMITY_THRESHOLD
    
    # Define rules
    rule1_tt = df['is_audio_interaction'].astype(bool) if 'is_audio_interaction' in df else False
    rule2_prox = (is_visual_confident & is_close)
    rule3_kcds = df['is_sustained_kcds'].astype(bool)
    rule4_ohs = df['is_sustained_ohs'].astype(bool)

    df['rule1_turn_taking'] = rule1_tt
    df['rule2_close_proximity'] = rule2_prox
    df['rule3_kcds_speaking'] = (rule3_kcds & is_visual_anchor)
    # ------------------------------------------------------------

    # 2. Initialize Default State: Alone (3)
    df['interaction_type'] = 3
    
    # 3. Apply 'Available' (2)
    # A frame is classified as Available if it meets the visual persistence criteria (person seen recently) or if there is sustained OHS presence, but does not meet the criteria for Interacting.
    available_mask = is_available_visual | (rule4_ohs & is_visual_anchor)
    df.loc[available_mask, 'interaction_type'] = 2
    
    # 4. Apply 'Interacting' (1)
    # A frame is classified as Interacting if it meets any of the following criteria:
    # - Turn-taking: Identified as part of an audio interaction bout based on the turn-taking logic.
    # - Close Proximity: The proximity feature exceeds the defined threshold, indicating close physical proximity to another person.
    # - Sustained KCDS: There is sustained CDS presence over the defined window, and the presence score meets the gating threshold, indicating a strong likelihood of interaction
    interacting_mask = (
        rule1_tt | 
        df['rule3_kcds_speaking'] | # Use the gated rule we just saved
        rule2_prox
    )
    df.loc[interacting_mask, 'interaction_type'] = 1

    # 5. Handle Binary Mode
    if social_state_mode == "binary":
        df.loc[df['interaction_type'] == 3, 'interaction_type'] = 2
        
    return df

def find_speech_segments(video_df: pd.DataFrame, column_name: str) -> List[Dict]:
    """
    Identifies continuous segments in a video DataFrame where the specified column is 1.
    Segments are defined as consecutive frames where the column is 1, allowing for gaps up to the sample rate (to account for the sampling of frames).
    Returns a list of dictionaries with 'start', 'end', and 'type' for each segment.
    
    Parameters
    ----------
    video_df: pd.DataFrame
        DataFrame containing frame-level data for a single video, indexed by frame_number.
    column_name: str
        The name of the column to analyze for segment detection (e.g., 'has_kchi').
    """
    segments = []

    if column_name not in video_df.columns:
        return segments

    speech_frames = video_df[video_df[column_name] == 1]
    if speech_frames.empty:
        return segments

    # Determine frame numbers: use index if frame_number isn't explicit
    if 'frame_number' in speech_frames.columns:
        frame_numbers = speech_frames['frame_number'].values.astype(int)
    else:
        frame_numbers = speech_frames.index.values.astype(int)

    if len(frame_numbers) < 2:
        # Handle the single-frame or single-segment case
        return [{
            'start': int(frame_numbers[0]),
            'end': int(frame_numbers[-1]),
            'type': column_name.split('_')[-1]
        }]

    # Vectorized gap calculation (difference between consecutive frames)
    gaps = np.diff(frame_numbers) 

    # Identify indices where the gap exceeds SAMPLE_RATE (segment breaks)
    # np.where returns a tuple, take the first element (the array of indices)
    break_indices = np.where(gaps > AnalysisConfig.SAMPLE_RATE)[0]
    
    # Segment boundaries reconstruction
    current_start = frame_numbers[0]
    
    # Loop over break indices
    for i in break_indices:
        segments.append({
            'start': int(current_start),
            'end': int(frame_numbers[i]),
            'type': column_name.split('_')[-1]
        })
        current_start = frame_numbers[i + 1]

    # Append last segment
    segments.append({
        'start': int(current_start),
        'end': int(frame_numbers[-1]),
        'type': column_name.split('_')[-1]
    })

    return segments

def check_audio_interaction_turn_taking(df: pd.DataFrame, fps: int) -> pd.Series:
    """
    Identifies continuous audio interaction bouts where KCHI and CDS segments 
    are linked by a small gap (<= MAX_TURN_TAKING_GAP_SEC) or inter-sperspeaker gap (<= MAX_SAME_SPEAKER_GAP_SEC).
    Only segments that contain both KCHI and CDS are classified as Interacting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain ['video_id', 'frame_number', 'has_kchi', 'has_cds']
    fps : int
        Frames per second (used for gap calculation)
    
    Returns
    -------
    pd.Series: Boolean Series of 'is_audio_interaction' for all frames.
    """
    if df is None or df.empty:
        return pd.Series(False, index=df.index if df is not None else [], name='is_audio_interaction')
    
    MAX_GAP_FRAMES = AnalysisConfig.MAX_TURN_TAKING_GAP_SEC * fps
    MAX_SAME_SPEAKER_GAP_FRAMES = AnalysisConfig.MAX_SAME_SPEAKER_GAP_SEC * fps
    all_results = []

    # Process each video separately
    for _, video_df in df.groupby('video_id'):
        video_df = video_df.copy() 
        video_df.set_index('frame_number', inplace=True) 
        video_df['is_audio_interaction'] = False
        
        # Identify KCHI and KCDS segments
        kchi_segments = find_speech_segments(video_df, 'has_kchi')
        kcds_segments = find_speech_segments(video_df, 'has_cds')
        all_segments = sorted(kchi_segments + kcds_segments, key=lambda x: x['start'])
        
        if not all_segments:
            video_df.reset_index(inplace=True)
            all_results.append(video_df[['frame_number', 'is_audio_interaction']])
            continue
            
        interaction_windows = []
        
        # Phase: Merge Segments and Filter for Dual Speakers
        current_window = {
            'start': all_segments[0]['start'],
            'end': all_segments[0]['end'],
            'types': {all_segments[0]['type']}
        }

        for seg in all_segments[1:]:
            is_same_type = seg['type'] in current_window['types']
            gap = seg['start'] - current_window['end']
            
            if is_same_type:
                # Prevent merging identical segments across gaps larger than allowed
                if gap > MAX_SAME_SPEAKER_GAP_FRAMES:
                    # Finalize current window (it's not turn-taking if it's only one speaker type)
                    if 'kchi' in current_window['types'] and 'cds' in current_window['types']:
                        interaction_windows.append(current_window)
                    
                    current_window = {'start': seg['start'], 'end': seg['end'], 'types': {seg['type']}}
                else:
                    current_window['end'] = seg['end']
                    current_window['types'].add(seg['type'])
            else: 
                if gap <= MAX_GAP_FRAMES:
                    current_window['end'] = seg['end']
                    current_window['types'].add(seg['type'])
                else:
                    if 'kchi' in current_window['types'] and 'cds' in current_window['types']:
                        interaction_windows.append(current_window)
                    
                    current_window = {'start': seg['start'], 'end': seg['end'], 'types': {seg['type']}}

        if 'kchi' in current_window['types'] and 'cds' in current_window['types']:
            interaction_windows.append(current_window)

        # Mark frames within the validated interaction windows
        for window in interaction_windows:
            audio_mask = (video_df.loc[window['start'] : window['end'], 'has_kchi'] == 1) | \
                         (video_df.loc[window['start'] : window['end'], 'has_cds'] == 1)
            video_df.loc[audio_mask.index[audio_mask], 'is_audio_interaction'] = True
        
        video_df.reset_index(inplace=True)
        all_results.append(video_df[['frame_number', 'video_id', 'is_audio_interaction']])

    result_df = pd.concat(all_results, ignore_index=True)
    return result_df['is_audio_interaction']

def main(db_path: Path, 
         output_dir: Path, 
         video_list: list = None, 
         social_state_mode: str = "tertiary"):
    """
    Orchestrates the frame-level processing pipeline.
    
    Parameters
    ----------
    db_path: Path
        Path to the SQLite database containing multimodal data.
    output_dir: Path
        Directory where the output CSV will be saved.
    video_list: list
        Optional list of video names to process. If None, processes all videos in the database.
    social_state_mode: str
        "binary" - classifies frames into Interacting vs Not-Interacting (Available + Alone)
        "tertiary" - classifies frames into Interacting, Available, Alone (default)
    """    
    with sqlite3.connect(db_path) as conn:
        prepare_persistent_tables(conn)
        
        all_data = get_all_analysis_data(conn, video_list)
        
        # Audio Interaction Logic (Turn-taking)
        all_data['is_audio_interaction'] = check_audio_interaction_turn_taking(all_data, DataConfig.FPS)
        
        # Calculate Features and Vectorized Classification
        all_data = calculate_window_features(all_data)
        all_data = classify_frames(all_data, social_state_mode=social_state_mode)
        
        output_path = output_dir / Analysis.FRAME_LEVEL_INTERACTIONS_CSV.name
        all_data.to_csv(output_path, index=False)
        logging.info(f"✅ Frame-level analysis saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vectorized Frame-Level Social Analysis")
    parser.add_argument('--social_state_mode', type=str, choices=['binary', 'tertiary'], default='tertiary')
    parser.add_argument('--video_list', type=str, nargs='+', default=None)
    args = parser.parse_args()
    
    main(db_path=Path(DataPaths.INFERENCE_DB_PATH), output_dir=Analysis.BASE_OUTPUT_DIR, video_list=args.video_list, social_state_mode=args.social_state_mode)
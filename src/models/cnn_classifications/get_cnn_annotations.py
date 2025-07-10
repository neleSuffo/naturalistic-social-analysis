import sqlite3
import os
import pandas as pd
import re
import random
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
from constants import DetectionPaths
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cnn_annotations.log'),
        logging.StreamHandler()
    ]
)

CATEGORY_IDS = [1,2,3,4,5,6,7,8,10,11,12]

def fetch_all_annotations(
    category_ids: List[int],
) -> List[tuple]:
    """
    This function fetches annotations from the database for specific category IDs.
    Supports fetching person annotations, object annotations, or both.

    Parameters
    ----------
    category_ids : List[int]
        The list of category IDs to filter the annotations
    
    Returns
    -------
    list of tuple
        The list of annotations.
    """
    logging.info(f"Starting annotation fetch for category IDs: {category_ids}")
    
    try:
        conn = sqlite3.connect(DetectionPaths.quantex_annotations_db_path)
        cursor = conn.cursor()
        logging.info(f"Successfully connected to database: {DetectionPaths.quantex_annotations_db_path}")
        
        placeholders = ", ".join("?" for _ in category_ids)
        object_target_class_ids = [3, 4, 5, 6, 7, 8, 12]

        # Construct conditional filter for object_interaction
        object_placeholders = ", ".join(str(x) for x in object_target_class_ids)
        
        query = f"""
        SELECT DISTINCT 
            a.category_id, 
            a.bbox, 
            a.object_interaction,
            i.file_name,
            a.gaze_directed_at_child,
            a.person_age
        FROM annotations a
        JOIN images i ON a.image_id = i.frame_id AND a.video_id = i.video_id
        JOIN videos v ON a.video_id = v.id
        WHERE a.category_id IN ({placeholders}) 
            AND a.outside = 0 
            AND v.file_name NOT LIKE '%id255237_2022_05_08_04%'
            AND (
                (a.category_id IN ({object_placeholders}) AND a.object_interaction = 'Yes') OR
                (a.category_id NOT IN ({object_placeholders}))
            )
            -- Exclude frames that have any annotation with category_id = -1
            AND NOT EXISTS (
                SELECT 1 FROM annotations a2
                WHERE a2.video_id = a.video_id
                AND a2.image_id = a.image_id
                AND a2.category_id = -1
            )
        ORDER BY a.video_id, a.image_id
        """
        
        logging.info("Executing annotation query...")
        cursor.execute(query, category_ids)
        annotations = cursor.fetchall()
        
        logging.info(f"Successfully fetched {len(annotations)} annotations")
        
        # Log unique category ids
        unique_category_ids = set(annotation[0] for annotation in annotations)
        logging.info(f"Found annotations for category IDs: {sorted(unique_category_ids)}")
        
        # Log distribution of annotations by category
        category_counts = {}
        for annotation in annotations:
            cat_id = annotation[0]
            category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
        
        logging.info("Category distribution:")
        for cat_id, count in sorted(category_counts.items()):
            logging.info(f"  Category {cat_id}: {count} annotations")
            
        conn.close()
        logging.info("Database connection closed")
        
        return annotations
        
    except Exception as e:
        logging.error(f"Error fetching annotations: {str(e)}")
        if 'conn' in locals():
            conn.close()
        raise
    
def convert_to_multilabel_format(input_csv_path, output_csv_path):
    """
    Convert annotations from category-based format to multi-label format.
    Combines multiple rows per frame into a single row with binary labels.
    
    Parameters
    ----------
    input_csv_path : str
        Path to the input CSV with category-based annotations
    output_csv_path : str
        Path to save the converted multi-label CSV
    """
    logging.info(f"Converting annotations from {input_csv_path} to multi-label format")
    
    try:
        # Load the annotations
        df = pd.read_csv(input_csv_path)
        logging.info(f"Loaded {len(df)} annotations from CSV")
        
        # Define category mappings
        # Based on your category IDs: [1,2,3,4,5,6,7,8,10,11,12]
        person_categories = [1, 2]
        face_categories = [10] 
        object_categories = [3, 4, 5, 6, 7, 8, 12]  # Objects
        
        # Initialize the result DataFrame
        unique_frames = df['file_name'].unique()
        logging.info(f"Found {len(unique_frames)} unique frames")
        
        result_data = []
        
        for frame_name in unique_frames:
            # Get all annotations for this frame
            frame_annotations = df[df['file_name'] == frame_name]
            
            # Initialize labels for this frame
            labels = {
                'file_name': frame_name,
                'adult_person_present': 0,
                'child_person_present': 0,
                'adult_face_present': 0,
                'child_face_present': 0,
                'object_interaction': 0
            }
            
            # Process each annotation for this frame
            for _, annotation in frame_annotations.iterrows():
                category_id = annotation['category_id']
                person_age = annotation.get('person_age', '')
                object_interaction = annotation.get('object_interaction', 'No')
                
                # Check for persons
                if category_id in person_categories:
                    if person_age and person_age.lower() == "adult":
                        labels['adult_person_present'] = 1
                    elif person_age and person_age.lower() == "child":
                        labels['child_person_present'] = 1
                    else:
                        continue  # Skip if person_age is not specified or unclear
                
                # Check for faces
                elif category_id in face_categories:
                    if person_age and person_age.lower() == "adult":
                        labels['adult_face_present'] = 1
                    elif person_age and person_age.lower() == "child":
                        labels['child_face_present'] = 1
                    else:
                        continue
                
                # Check for object interaction
                elif category_id in object_categories:
                    if object_interaction and object_interaction.lower() == 'yes':
                        labels['object_interaction'] = 1
            
            result_data.append(labels)
        
        # Create result DataFrame
        result_df = pd.DataFrame(result_data)
        
        # Log statistics
        logging.info("=== Conversion Statistics ===")
        logging.info(f"Total frames processed: {len(result_df)}")
        logging.info(f"Adult person present: {result_df['adult_person_present'].sum()}")
        logging.info(f"Child person present: {result_df['child_person_present'].sum()}")
        logging.info(f"Adult face present: {result_df['adult_face_present'].sum()}")
        logging.info(f"Child face present: {result_df['child_face_present'].sum()}")
        logging.info(f"Object interaction: {result_df['object_interaction'].sum()}")
        
        # Log label distribution
        total_frames = len(result_df)
        logging.info("Label distribution (percentage):")
        for col in ['adult_person_present', 'child_person_present', 'adult_face_present', 'child_face_present', 'object_interaction']:
            percentage = (result_df[col].sum() / total_frames) * 100
            logging.info(f"  {col}: {percentage:.1f}%")
        
        # Save to CSV
        result_df.to_csv(output_csv_path, index=False)
        logging.info(f"Converted annotations saved to {output_csv_path}")
        
        return result_df
        
    except Exception as e:
        logging.error(f"Error during conversion: {str(e)}")
        raise
    
def extract_video_filename(frame_id: str) -> str:
    """Extract video filename from frame_id."""
    try:
        result = re.match(r'(.*)_\d{6}\.jpg', frame_id).group(1) + '.mp4'
        logging.debug(f"Extracted video filename '{result}' from frame_id '{frame_id}'")
        return result
    except AttributeError:
        logging.error(f"Could not extract video filename from frame_id: {frame_id}")
        raise

def augment_with_random_frames(csv_path, output_path):
    """Augment annotations with random frames from the same videos."""
    logging.info(f"Starting augmentation process...")
    logging.info(f"Input CSV: {csv_path}")
    logging.info(f"Output CSV: {output_path}")
    
    try:
        # Load existing annotations
        logging.info("Loading existing annotations from CSV...")
        df = pd.read_csv(csv_path)
        logging.info(f"Loaded {len(df)} existing annotations")

        # Extract video filenames
        logging.info("Extracting video filenames from frame IDs...")
        df['video_filename'] = df['file_name'].apply(extract_video_filename)
        unique_videos = df['video_filename'].unique()
        logging.info(f"Found {len(unique_videos)} unique videos")

        conn = sqlite3.connect(DetectionPaths.quantex_annotations_db_path)
        cursor = conn.cursor()
        logging.info("Connected to database for frame augmentation")

        all_new_rows = []
        processed_videos = 0

        for video_filename in unique_videos:
            logging.info(f"Processing video {processed_videos + 1}/{len(unique_videos)}: {video_filename}")
            
            # Get internal DB id
            cursor.execute("SELECT id FROM videos WHERE file_name = ?", (video_filename,))
            video_row = cursor.fetchone()
            if not video_row:
                logging.warning(f"Video '{video_filename}' not found in database")
                continue
            video_db_id = video_row[0]
            logging.debug(f"Video DB ID: {video_db_id}")

            # Get max frame number from images table
            cursor.execute("SELECT MAX(frame_id) FROM images WHERE video_id = ?", (video_db_id,))
            max_frame = cursor.fetchone()[0]
            if max_frame is None:
                logging.warning(f"No frames found for video '{video_filename}'")
                continue
            
            logging.debug(f"Max frame number: {max_frame}")

            # Get annotated frames for this video
            video_annotations = df[df['video_filename'] == video_filename]
            annotated_frame_numbers = set()
            
            for _, row in video_annotations.iterrows():
                # Extract frame number from file_name
                frame_name = row['file_name']
                try:
                    # Extract frame number from format: video_name_XXXXXX.jpg
                    frame_match = re.search(r'_(\d{6})\.jpg$', frame_name)
                    if frame_match:
                        frame_number = int(frame_match.group(1))
                        annotated_frame_numbers.add(frame_number)
                except Exception as e:
                    logging.warning(f"Could not extract frame number from {frame_name}: {e}")
                    continue
            
            logging.debug(f"Found {len(annotated_frame_numbers)} annotated frames: {sorted(annotated_frame_numbers)}")

            # Collect all frame numbers divisible by 30 that are NOT already annotated
            all_frames_div_30 = [f for f in range(0, max_frame + 1) if f % 30 == 0]
            valid_frames = [f for f in all_frames_div_30 if f not in annotated_frame_numbers]
            
            logging.debug(f"Total frames divisible by 30: {len(all_frames_div_30)}")
            logging.debug(f"Valid unannotated frames: {len(valid_frames)}")

            # Count annotated frames for this video
            n_annotated = len(annotated_frame_numbers)
            logging.debug(f"Number of annotated frames: {n_annotated}")

            # Sample half of the annotated frames, or all if less than 10
            n_sample = int((min(n_annotated, len(valid_frames))) / 4)
            if n_sample == 0:
                logging.warning(f"No unannotated frames to sample for video '{video_filename}'")
                continue
                
            sampled_frames = random.sample(valid_frames, n_sample)
            logging.debug(f"Sampling {n_sample} random frames from {len(valid_frames)} available")
            logging.debug(f"Sampled frames: {sorted(sampled_frames)}")

            for fn in sampled_frames:
                file_name = f"{video_filename.replace('.mp4', '')}_{str(fn).zfill(6)}.jpg"
                all_new_rows.append({
                    'file_name': file_name,
                    'adult_person_present': 0,
                    'child_person_present': 0,
                    'adult_face_present': 0,
                    'child_face_present': 0,
                    'object_interaction': 0
                })
            
            processed_videos += 1
            if processed_videos % 10 == 0:
                logging.info(f"Processed {processed_videos}/{len(unique_videos)} videos")

        logging.info(f"Generated {len(all_new_rows)} new random frame annotations")

        # Combine original with new rows
        df = df.drop(columns=['video_filename'])
        df_augmented = pd.concat([df, pd.DataFrame(all_new_rows)], ignore_index=True)
        df_augmented = df_augmented.sort_values(by='file_name').reset_index(drop=True)

        logging.info(f"Combined dataset size: {len(df_augmented)} annotations")
        logging.info(f"Original annotations: {len(df)}")
        logging.info(f"Added random frames: {len(all_new_rows)}")

        # Verify no duplicates
        duplicates = df_augmented['file_name'].duplicated().sum()
        if duplicates > 0:
            logging.warning(f"Found {duplicates} duplicate file names in augmented dataset!")
        else:
            logging.info("No duplicate file names found in augmented dataset")

        # Save
        df_augmented.to_csv(output_path, index=False)
        logging.info(f"Augmented CSV saved to {output_path}")

        conn.close()
        logging.info("Database connection closed")
        
    except Exception as e:
        logging.error(f"Error during augmentation: {str(e)}")
        if 'conn' in locals():
            conn.close()
        raise

def extract_child_id(frame_name):
    """Extract child ID from frame filename."""
    try:
        # Assuming format: quantex_at_home_idXXXXXX_YYYY_MM_DD_HH_XXXXXX.jpg
        match = re.search(r'id(\d+)_', frame_name)
        if match:
            return match.group(1)
        else:
            logging.warning(f"Could not extract child ID from {frame_name}")
            return None
    except Exception as e:
        logging.error(f"Error extracting child ID from {frame_name}: {e}")
        return None

def create_stratified_splits_by_child(csv_path, output_dir, test_size=0.1, val_size=0.1, random_state=42):
    """
    Create stratified train/val/test splits using up to three videos per child.
    This balances having sufficient data with preventing overfitting to specific 
    children's characteristics. Videos are selected based on frame count (highest first).
    
    Parameters
    ----------
    csv_path : str
        Path to the multi-label annotations CSV
    output_dir : str
        Directory to save the split CSV files
    test_size : float
        Proportion of data for test set
    val_size : float
        Proportion of data for validation set
    random_state : int
        Random seed for reproducibility
    """
    logging.info(f"Creating stratified splits by child (up to 3 videos per child) from {csv_path}")
    logging.info(f"Target split: {(1-test_size-val_size)*100:.1f}% train, {val_size*100:.1f}% val, {test_size*100:.1f}% test")
    
    try:
        # Load the data
        df = pd.read_csv(csv_path)
        logging.info(f"Loaded {len(df)} annotations")
        
        # Extract child IDs and video names
        df['child_id'] = df['file_name'].apply(extract_child_id)
        df['video_name'] = df['file_name'].apply(lambda x: extract_video_filename(x))
        df = df.dropna(subset=['child_id'])  # Remove rows where child_id couldn't be extracted
        logging.info(f"Retained {len(df)} annotations after child ID extraction")
        
        # Analyze videos per child
        child_video_counts = df.groupby('child_id')['video_name'].nunique().to_dict()
        video_frame_counts = df.groupby(['child_id', 'video_name']).size().to_dict()
        
        logging.info("=== Child-Video Analysis ===")
        total_children = len(child_video_counts)
        children_with_multiple_videos = sum(1 for count in child_video_counts.values() if count > 1)
        
        logging.info(f"Total children: {total_children}")
        logging.info(f"Children with multiple videos: {children_with_multiple_videos}")
        logging.info(f"Children with single video: {total_children - children_with_multiple_videos}")
        
        # Log detailed video distribution
        for child_id in sorted(child_video_counts.keys()):
            video_count = child_video_counts[child_id]
            child_videos = df[df['child_id'] == child_id]['video_name'].unique()
            logging.info(f"Child {child_id}: {video_count} videos")
            for video in child_videos:
                frame_count = video_frame_counts.get((child_id, video), 0)
                logging.info(f"  {video}: {frame_count} frames")
        
        # Select up to three videos per child (ordered by frame count)
        np.random.seed(random_state)
        selected_child_videos = {}
        
        for child_id in child_video_counts.keys():
            child_videos = df[df['child_id'] == child_id]['video_name'].unique()
            
            # Calculate frame counts for all videos of this child
            video_frame_counts_for_child = {}
            for video in child_videos:
                frame_count = len(df[(df['child_id'] == child_id) & (df['video_name'] == video)])
                video_frame_counts_for_child[video] = frame_count
            
            # Sort videos by frame count (descending) and select up to 3
            sorted_videos = sorted(video_frame_counts_for_child.items(), key=lambda x: x[1], reverse=True)
            selected_videos = [video for video, _ in sorted_videos[:3]]  # Take up to 3 best videos
            
            selected_child_videos[child_id] = selected_videos
            
            if len(child_videos) == 1:
                logging.info(f"Child {child_id}: Only 1 video available - {selected_videos[0]} ({video_frame_counts_for_child[selected_videos[0]]} frames)")
            else:
                total_selected_frames = sum(video_frame_counts_for_child[v] for v in selected_videos)
                logging.info(f"Child {child_id}: Selected {len(selected_videos)} videos (total: {total_selected_frames} frames)")
                for i, video in enumerate(selected_videos):
                    logging.info(f"  {i+1}. {video}: {video_frame_counts_for_child[video]} frames")
                
                # Log excluded videos if any
                excluded_videos = [(v, c) for v, c in video_frame_counts_for_child.items() if v not in selected_videos]
                if excluded_videos:
                    excluded_frames = sum(c for _, c in excluded_videos)
                    logging.info(f"  (Excluded {len(excluded_videos)} videos with {excluded_frames} frames: {excluded_videos})")
        
        # Filter dataset to only include selected videos
        def keep_frame(row):
            child_id = row['child_id']
            video_name = row['video_name']
            return child_id in selected_child_videos and video_name in selected_child_videos[child_id]
        
        original_size = len(df)
        df = df[df.apply(keep_frame, axis=1)].copy()
        filtered_size = len(df)
        
        logging.info(f"Filtered dataset to one video per child:")
        logging.info(f"  Original size: {original_size} frames")
        logging.info(f"  Filtered size: {filtered_size} frames")
        logging.info(f"  Removed: {original_size - filtered_size} frames ({((original_size - filtered_size)/original_size)*100:.1f}%)")
        
        # Verify we have at most three videos per child
        remaining_child_video_counts = df.groupby('child_id')['video_name'].nunique().to_dict()
        for child_id, video_count in remaining_child_video_counts.items():
            if video_count > 3:
                logging.error(f"Child {child_id} still has {video_count} videos after filtering (expected max 3)!")
                raise ValueError(f"Filtering failed: Child {child_id} has {video_count} videos (expected max 3)")
        
        # Log the distribution of video counts per child
        video_count_distribution = {}
        for child_id, video_count in remaining_child_video_counts.items():
            video_count_distribution[video_count] = video_count_distribution.get(video_count, 0) + 1
        
        logging.info("Video count distribution per child after filtering:")
        for num_videos, num_children in sorted(video_count_distribution.items()):
            logging.info(f"  {num_children} children with {num_videos} video(s)")
        
        logging.info(f"✓ Verification passed: All {len(remaining_child_video_counts)} children have ≤3 videos")
        
        # Filter to keep only frames that are multiples of 30
        def is_frame_multiple_of_30(file_name):
            """Check if frame number is multiple of 30."""
            try:
                # Extract frame number from format: video_name_XXXXXX.jpg
                frame_match = re.search(r'_(\d{6})\.jpg$', file_name)
                if frame_match:
                    frame_number = int(frame_match.group(1))
                    return frame_number % 30 == 0
                return False
            except Exception:
                return False
        
        # Apply frame filtering
        pre_frame_filter_count = len(df)
        df['is_multiple_30'] = df['file_name'].apply(is_frame_multiple_of_30)
        df = df[df['is_multiple_30']].drop(columns=['is_multiple_30'])
        post_frame_filter_count = len(df)
        removed_by_frame_filter = pre_frame_filter_count - post_frame_filter_count
        
        logging.info(f"Frame filtering results (after video selection):")
        logging.info(f"  Before frame filtering: {pre_frame_filter_count} frames")
        logging.info(f"  After frame filtering: {post_frame_filter_count} frames")
        logging.info(f"  Removed by frame filter: {removed_by_frame_filter} frames ({(removed_by_frame_filter/pre_frame_filter_count)*100:.1f}%)")
        
        if post_frame_filter_count == 0:
            raise ValueError("No frames remain after filtering for multiples of 30")
        
        # Log some examples of kept frames
        sample_frames = df['file_name'].head(5).tolist()
        logging.info(f"Sample kept frames: {sample_frames}")
        
        # Get unique child IDs and their frame counts (after all filtering)
        unique_children = df['child_id'].unique()
        logging.info(f"Final children count: {len(unique_children)} children: {sorted(unique_children)}")
        
        # Calculate frame counts per child (after filtering)
        child_frame_counts = df['child_id'].value_counts().to_dict()
        total_frames = len(df)
        
        logging.info("=== Final Child Frame Distribution (after all filtering) ===")
        for child_id in sorted(unique_children):
            count = child_frame_counts[child_id]
            percentage = (count / total_frames) * 100
            selected_videos = selected_child_videos[child_id]
            video_str = ", ".join(selected_videos) if len(selected_videos) > 1 else selected_videos[0]
            logging.info(f"Child {child_id} ({video_str}): {count} frames ({percentage:.1f}%)")
        
        # Verify minimum frame requirements
        min_frames_per_child = min(child_frame_counts.values())
        if min_frames_per_child < 10:
            logging.warning(f"Some children have very few frames (minimum: {min_frames_per_child})")
            # Log children with few frames
            few_frames_children = [child_id for child_id, count in child_frame_counts.items() if count < 20]
            if few_frames_children:
                logging.warning(f"Children with < 20 frames: {few_frames_children}")
        
        # Calculate class distribution per child
        label_columns = ['adult_person_present', 'child_person_present', 'adult_face_present', 
                        'child_face_present', 'object_interaction']
        
        child_stats = {}
        for child_id in unique_children:
            child_data = df[df['child_id'] == child_id]
            stats = {
                'n_frames': len(child_data),
                'class_counts': {}
            }
            for col in label_columns:
                positive_count = child_data[col].sum()
                stats['class_counts'][col] = {
                    'positive': positive_count,
                    'negative': len(child_data) - positive_count,
                    'positive_ratio': positive_count / len(child_data)
                }
            child_stats[child_id] = stats
        
        # Create stratification key for each child based on their dominant class patterns
        def create_stratification_key(child_id):
            stats = child_stats[child_id]
            key = []
            for col in label_columns:
                ratio = stats['class_counts'][col]['positive_ratio']
                key.append(1 if ratio > 0.1 else 0)
            return tuple(key)
        
        child_strat_keys = {child_id: create_stratification_key(child_id) for child_id in unique_children}
        
        # Group children by stratification key
        strat_groups = defaultdict(list)
        for child_id, key in child_strat_keys.items():
            strat_groups[key].append(child_id)
        
        logging.info(f"Created {len(strat_groups)} stratification groups:")
        for key, children in strat_groups.items():
            total_frames_in_group = sum(child_frame_counts[child] for child in children)
            logging.info(f"  Group {key}: {len(children)} children, {total_frames_in_group} frames")
        
        # Initialize splits
        train_children = []
        val_children = []
        test_children = []
        
        train_frames = 0
        val_frames = 0
        test_frames = 0
        
        target_train_frames = total_frames * (1 - test_size - val_size)
        target_val_frames = total_frames * val_size
        target_test_frames = total_frames * test_size
        
        np.random.seed(random_state)
        
        # Sort all children by frame count (ascending) to handle smaller children first
        # This helps with better distribution across splits
        children_by_frames = sorted(unique_children, key=lambda x: child_frame_counts[x])
        
        # First pass: assign children to splits using a round-robin approach 
        # weighted by current deficit
        for i, child_id in enumerate(children_by_frames):
            child_frames = child_frame_counts[child_id]
            
            # Calculate current percentages
            current_train_pct = train_frames / total_frames if total_frames > 0 else 0
            current_val_pct = val_frames / total_frames if total_frames > 0 else 0
            current_test_pct = test_frames / total_frames if total_frames > 0 else 0
            
            # Calculate deficits (how far we are from target)
            train_deficit = (1 - test_size - val_size) - current_train_pct
            val_deficit = val_size - current_val_pct
            test_deficit = test_size - current_test_pct
            
            # Calculate what percentage each split would have after adding this child
            train_pct_after = (train_frames + child_frames) / total_frames
            val_pct_after = (val_frames + child_frames) / total_frames
            test_pct_after = (test_frames + child_frames) / total_frames
            
            # Decide assignment based on largest deficit and avoiding overshooting
            max_acceptable_overshoot = 0.05  # Allow 5% overshoot
            
            # Create list of viable options (split_name, deficit, pct_after)
            options = []
            if train_pct_after <= (1 - test_size - val_size) + max_acceptable_overshoot:
                options.append(('train', train_deficit, train_pct_after))
            if val_pct_after <= val_size + max_acceptable_overshoot:
                options.append(('val', val_deficit, val_pct_after))
            if test_pct_after <= test_size + max_acceptable_overshoot:
                options.append(('test', test_deficit, test_pct_after))
            
            # If no options are viable (all would overshoot), allow assignment to train
            if not options:
                options = [('train', train_deficit, train_pct_after)]
            
            # Sort by deficit (descending) - assign to split with largest deficit
            options.sort(key=lambda x: x[1], reverse=True)
            assignment = options[0][0]
            
            if assignment == 'train':
                train_children.append(child_id)
                train_frames += child_frames
            elif assignment == 'val':
                val_children.append(child_id)
                val_frames += child_frames
            else:  # test
                test_children.append(child_id)
                test_frames += child_frames
            
            logging.debug(f"Assigned child {child_id} ({child_frames} frames) to {assignment}")
            logging.debug(f"  Current percentages: train={current_train_pct:.3f}, val={current_val_pct:.3f}, test={current_test_pct:.3f}")
            logging.debug(f"  Deficits: train={train_deficit:.3f}, val={val_deficit:.3f}, test={test_deficit:.3f}")
        
        # Second pass: rebalance if any split is significantly under-represented
        # Move children from over-represented splits to under-represented ones
        max_iterations = 3
        for iteration in range(max_iterations):
            current_train_pct = train_frames / total_frames
            current_val_pct = val_frames / total_frames  
            current_test_pct = test_frames / total_frames
            
            target_train_pct = 1 - test_size - val_size
            target_val_pct = val_size
            target_test_pct = test_size
            
            # Check if validation set is significantly under target
            if current_val_pct < target_val_pct - 0.03 and len(train_children) > 1:  # 3% threshold
                # Find smallest child in train that could be moved to val
                candidate = min(train_children, key=lambda x: child_frame_counts[x])
                candidate_frames = child_frame_counts[candidate]
                new_val_pct = (val_frames + candidate_frames) / total_frames
                
                if new_val_pct <= target_val_pct + 0.05:  # Don't overshoot by more than 5%
                    train_children.remove(candidate)
                    val_children.append(candidate)
                    train_frames -= candidate_frames
                    val_frames += candidate_frames
                    logging.info(f"Rebalancing: moved child {candidate} from train to val")
                    continue
            
            # Check if test set is significantly under target
            if current_test_pct < target_test_pct - 0.03 and len(train_children) > 1:
                candidate = min(train_children, key=lambda x: child_frame_counts[x])
                candidate_frames = child_frame_counts[candidate]
                new_test_pct = (test_frames + candidate_frames) / total_frames
                
                if new_test_pct <= target_test_pct + 0.05:
                    train_children.remove(candidate)
                    test_children.append(candidate)
                    train_frames -= candidate_frames
                    test_frames += candidate_frames
                    logging.info(f"Rebalancing: moved child {candidate} from train to test")
                    continue
            
            # If no rebalancing was needed, break
            break
        
        # Final safety check: ensure each split has at least two children
        min_children_per_split = 2
        
        # Ensure validation set has at least 2 children
        while len(val_children) < min_children_per_split and len(train_children) > min_children_per_split:
            smallest_train_child = min(train_children, key=lambda x: child_frame_counts[x])
            train_children.remove(smallest_train_child)
            val_children.append(smallest_train_child)
            train_frames -= child_frame_counts[smallest_train_child]
            val_frames += child_frame_counts[smallest_train_child]
            logging.info(f"Safety: moved child {smallest_train_child} from train to val (val now has {len(val_children)} children)")
        
        # Ensure test set has at least 2 children
        while len(test_children) < min_children_per_split and len(train_children) > min_children_per_split:
            smallest_train_child = min(train_children, key=lambda x: child_frame_counts[x])
            train_children.remove(smallest_train_child)
            test_children.append(smallest_train_child)
            train_frames -= child_frame_counts[smallest_train_child]
            test_frames += child_frame_counts[smallest_train_child]
            logging.info(f"Safety: moved child {smallest_train_child} from train to test (test now has {len(test_children)} children)")
        
        # Final verification
        if len(val_children) < min_children_per_split:
            logging.warning(f"Validation set only has {len(val_children)} children (target: {min_children_per_split})")
        if len(test_children) < min_children_per_split:
            logging.warning(f"Test set only has {len(test_children)} children (target: {min_children_per_split})")
        if len(train_children) < min_children_per_split:
            logging.warning(f"Train set only has {len(train_children)} children (target: {min_children_per_split})")
        
        logging.info(f"Final child split: {len(train_children)} train, {len(val_children)} val, {len(test_children)} test children")
        
        # Create dataframes for each split
        train_df = df[df['child_id'].isin(train_children)].copy()
        val_df = df[df['child_id'].isin(val_children)].copy()
        test_df = df[df['child_id'].isin(test_children)].copy()
        
        # Remove temporary columns from final datasets
        train_df = train_df.drop(columns=['child_id', 'video_name'])
        val_df = val_df.drop(columns=['child_id', 'video_name'])
        test_df = test_df.drop(columns=['child_id', 'video_name'])
        
        # Log split statistics
        train_pct = len(train_df) / total_frames * 100
        val_pct = len(val_df) / total_frames * 100
        test_pct = len(test_df) / total_frames * 100
        
        logging.info("=== Split Statistics ===")
        logging.info(f"Train: {len(train_df)} frames from {len(train_children)} children ({train_pct:.1f}%)")
        logging.info(f"Val: {len(val_df)} frames from {len(val_children)} children ({val_pct:.1f}%)")
        logging.info(f"Test: {len(test_df)} frames from {len(test_children)} children ({test_pct:.1f}%)")
        
        # Log which children went where (including selected video info)
        logging.info("=== Child Assignment Details (with selected videos) ===")
        for split_name, children in [('Train', train_children), ('Val', val_children), ('Test', test_children)]:
            total_split_frames = sum(child_frame_counts[child] for child in children)
            logging.info(f"{split_name} children:")
            for child in children:
                frames = child_frame_counts[child]
                selected_videos = selected_child_videos[child]
                video_str = ", ".join(selected_videos) if len(selected_videos) > 1 else selected_videos[0]
                logging.info(f"  Child {child} ({video_str}): {frames} frames")
        
        # Verify proportions are reasonable
        if abs(train_pct - (1-test_size-val_size)*100) > 15:
            logging.warning(f"Train set percentage ({train_pct:.1f}%) is significantly off target ({(1-test_size-val_size)*100:.1f}%)")
        if abs(val_pct - val_size*100) > 10:
            logging.warning(f"Val set percentage ({val_pct:.1f}%) is significantly off target ({val_size*100:.1f}%)")
        if abs(test_pct - test_size*100) > 10:
            logging.warning(f"Test set percentage ({test_pct:.1f}%) is significantly off target ({test_size*100:.1f}%)")
        
        # Log class distributions per split
        for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            logging.info(f"\n{split_name} class distribution:")
            for col in label_columns:
                positive = split_df[col].sum()
                total = len(split_df)
                ratio = positive / total if total > 0 else 0
                logging.info(f"  {col}: {positive}/{total} ({ratio:.3f})")
        
        # Save splits
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        train_path = os.path.join(output_dir, 'train_annotations.csv')
        val_path = os.path.join(output_dir, 'val_annotations.csv')
        test_path = os.path.join(output_dir, 'test_annotations.csv')
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logging.info(f"Saved splits to:")
        logging.info(f"  Train: {train_path}")
        logging.info(f"  Val: {val_path}")
        logging.info(f"  Test: {test_path}")
        
        # Final validation: verify all frames are multiples of 30
        def validate_frame_multiples_30(df, split_name):
            """Validate that all frames in the dataframe are multiples of 30."""
            invalid_frames = []
            for _, row in df.iterrows():
                file_name = row['file_name']
                try:
                    frame_match = re.search(r'_(\d{6})\.jpg$', file_name)
                    if frame_match:
                        frame_number = int(frame_match.group(1))
                        if frame_number % 30 != 0:
                            invalid_frames.append((file_name, frame_number))
                    else:
                        invalid_frames.append((file_name, "could_not_extract"))
                except Exception as e:
                    invalid_frames.append((file_name, f"error: {e}"))
            
            if invalid_frames:
                logging.error(f"{split_name} split contains {len(invalid_frames)} frames that are NOT multiples of 30:")
                for file_name, frame_info in invalid_frames[:5]:  # Show first 5
                    logging.error(f"  {file_name} -> {frame_info}")
                if len(invalid_frames) > 5:
                    logging.error(f"  ... and {len(invalid_frames) - 5} more")
                return False
            else:
                logging.info(f"{split_name} split validation: All {len(df)} frames are multiples of 30 ✓")
                return True
        
        # Validate all splits
        train_valid = validate_frame_multiples_30(train_df, "Train")
        val_valid = validate_frame_multiples_30(val_df, "Val") 
        test_valid = validate_frame_multiples_30(test_df, "Test")
        
        if not all([train_valid, val_valid, test_valid]):
            raise ValueError("Some splits contain frames that are not multiples of 30!")
        
        logging.info("✓ Final validation passed: All frames in all splits are multiples of 30")
        
        # Calculate class distribution statistics for each split
        def calculate_split_class_stats(split_df, split_name):
            """Calculate class distribution statistics for a split."""
            stats = {}
            total_samples = len(split_df)
            
            for col in label_columns:
                positive_count = split_df[col].sum()
                negative_count = total_samples - positive_count
                positive_ratio = positive_count / total_samples if total_samples > 0 else 0
                negative_ratio = negative_count / total_samples if total_samples > 0 else 0
                
                stats[col] = {
                    'positive_count': int(positive_count),
                    'negative_count': int(negative_count),
                    'total_count': int(total_samples),
                    'positive_ratio': float(positive_ratio),
                    'negative_ratio': float(negative_ratio),
                    'positive_percentage': float(positive_ratio * 100),
                    'negative_percentage': float(negative_ratio * 100)
                }
            
            return stats
        
        # Calculate statistics for each split
        train_class_stats = calculate_split_class_stats(train_df, "Train")
        val_class_stats = calculate_split_class_stats(val_df, "Val")
        test_class_stats = calculate_split_class_stats(test_df, "Test")
        
        # Log the class distribution statistics
        logging.info("\n=== Detailed Class Distribution Statistics ===")
        for split_name, split_stats in [('Train', train_class_stats), ('Val', val_class_stats), ('Test', test_class_stats)]:
            logging.info(f"\n{split_name} split class distribution:")
            for col, stats in split_stats.items():
                logging.info(f"  {col}:")
                logging.info(f"    Positive: {stats['positive_count']}/{stats['total_count']} ({stats['positive_percentage']:.1f}%)")
                logging.info(f"    Negative: {stats['negative_count']}/{stats['total_count']} ({stats['negative_percentage']:.1f}%)")
        
        # Save child assignment info for reference
        child_assignment = {
            'train': train_children,
            'val': val_children,
            'test': test_children,
            'train_frames': int(len(train_df)),
            'val_frames': int(len(val_df)),
            'test_frames': int(len(test_df)),
            'train_percentage': float(train_pct),
            'val_percentage': float(val_pct),
            'test_percentage': float(test_pct),
            'class_distribution': {
                'train': train_class_stats,
                'val': val_class_stats,
                'test': test_class_stats
            },
            'labels': label_columns,
            'video_selection_info': {
                'strategy': 'up_to_three_videos_per_child',
                'selection_method': 'most_frames_first',
                'max_videos_per_child': 3,
                'selected_child_videos': selected_child_videos,
                'original_total_frames': original_size,
                'after_video_selection_frames': filtered_size,
                'frames_removed_by_video_selection': original_size - filtered_size,
                'video_selection_reduction_percentage': ((original_size - filtered_size) / original_size) * 100,
                'children_with_multiple_videos_original': children_with_multiple_videos,
                'total_children': total_children
            },
            'split_creation_info': {
                'test_size': test_size,
                'val_size': val_size,
                'random_state': random_state,
                'total_children_final': len(unique_children),
                'total_frames_after_all_filtering': total_frames,
                'frames_removed_by_frame_filter': removed_by_frame_filter,
                'frame_filter_reduction_percentage': (removed_by_frame_filter / pre_frame_filter_count) * 100 if pre_frame_filter_count > 0 else 0,
                'minimum_frames_per_child': min_frames_per_child,
                'approach': 'up_to_three_videos_per_child_then_stratified_split'
            }
        }
        
        import json
        assignment_path = os.path.join(output_dir, 'child_split_assignment.json')
        with open(assignment_path, 'w') as f:
            json.dump(child_assignment, f, indent=2)
        logging.info(f"Saved child assignment to: {assignment_path}")
        
        return train_df, val_df, test_df, child_assignment
        
    except Exception as e:
        logging.error(f"Error creating stratified splits: {str(e)}")
        raise
 
def balance_dataset_classes(df, split_name, target_min_ratio=0.15, max_reduction=0.5, random_state=42):
    """
    Balance classes in a multi-label dataset using a more sophisticated approach.
    Instead of trying to achieve specific ratios, focuses on ensuring minimum representation
    for all classes while being aware of multi-label dependencies.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset to balance
    split_name : str
        Name of the split (for logging)
    target_min_ratio : float
        Minimum ratio for the minority class (default 0.15 = 15%)
    max_reduction : float
        Maximum allowed reduction in dataset size (default 0.5 = 50%)
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        Balanced dataset
    """
    logging.info(f"Balancing {split_name} dataset using multi-label aware approach...")
    logging.info(f"Target: Ensure at least {target_min_ratio*100:.1f}% representation for minority class in each label")
    logging.info(f"Maximum allowed reduction: {max_reduction*100:.1f}%")
    
    np.random.seed(random_state)
    
    label_columns = ['adult_person_present', 'child_person_present', 'adult_face_present', 
                    'child_face_present', 'object_interaction']
    
    # Start with a copy of the original dataframe
    balanced_df = df.copy()
    original_size = len(df)
    
    # Safety check: if dataset is too small, skip balancing
    if original_size < 20:
        logging.warning(f"{split_name} dataset is too small ({original_size} samples) - skipping balancing")
        return balanced_df
    
    # Log original class distribution and identify severely imbalanced classes
    logging.info(f"Original {split_name} class distribution:")
    severely_imbalanced_classes = []
    
    for col in label_columns:
        positive = balanced_df[col].sum()
        total = len(balanced_df)
        negative = total - positive
        positive_ratio = positive / total if total > 0 else 0
        negative_ratio = negative / total if total > 0 else 0
        
        # Identify which class is minority
        minority_ratio = min(positive_ratio, negative_ratio)
        minority_class = "positive" if positive_ratio < negative_ratio else "negative"
        
        logging.info(f"  {col}: {positive} pos ({positive_ratio:.3f}), {negative} neg ({negative_ratio:.3f})")
        logging.info(f"    Minority class: {minority_class} ({minority_ratio:.3f})")
        
        # Mark severely imbalanced classes (minority < target_min_ratio)
        # But only if we have enough minority samples to work with
        minority_count = positive if minority_class == "positive" else negative
        if minority_ratio < target_min_ratio and minority_count >= 3:  # Need at least 3 minority samples
            severely_imbalanced_classes.append({
                'column': col,
                'minority_class': minority_class,
                'minority_ratio': minority_ratio,
                'minority_count': minority_count,
                'majority_count': negative if minority_class == "positive" else positive
            })
            logging.warning(f"    {col} is severely imbalanced: {minority_class} class only {minority_ratio:.3f}")
        elif minority_count < 3:
            logging.warning(f"    {col} has too few minority samples ({minority_count}) - skipping balancing")
    
    if not severely_imbalanced_classes:
        logging.info(f"✓ No severely imbalanced classes found that can be balanced. Dataset is adequately balanced.")
        return balanced_df
    
    logging.info(f"\nFound {len(severely_imbalanced_classes)} severely imbalanced classes:")
    for class_info in severely_imbalanced_classes:
        logging.info(f"  {class_info['column']}: {class_info['minority_class']} class ({class_info['minority_ratio']:.3f})")
    
    # Strategy: Focus on the most imbalanced class first
    # Sort by minority ratio (most imbalanced first)
    severely_imbalanced_classes.sort(key=lambda x: x['minority_ratio'])
    
    total_removed = 0
    max_allowed_removal = int(original_size * max_reduction)
    
    for class_info in severely_imbalanced_classes:
        col = class_info['column']
        minority_class = class_info['minority_class']
        minority_count = class_info['minority_count']
        majority_count = class_info['majority_count']
        current_total = len(balanced_df)
        
        logging.info(f"\nBalancing {col} (minority: {minority_class})...")
        
        # Calculate how many majority samples to remove to reach target ratio
        # target_min_ratio = minority_count / new_total
        # new_total = minority_count / target_min_ratio
        target_total = int(minority_count / target_min_ratio)
        samples_to_remove = current_total - target_total
        
        # Limit removal to respect max_reduction constraint
        remaining_removal_budget = max_allowed_removal - total_removed
        samples_to_remove = min(samples_to_remove, remaining_removal_budget)
        
        # Don't remove more than 80% of majority class to avoid extreme skew
        max_majority_removal = int(majority_count * 0.8)
        samples_to_remove = min(samples_to_remove, max_majority_removal)
        
        logging.info(f"  Current: {minority_count} minority, {majority_count} majority, {current_total} total")
        logging.info(f"  Target total: {target_total} (to achieve {target_min_ratio:.3f} minority ratio)")
        logging.info(f"  Need to remove: {current_total - target_total} samples")
        logging.info(f"  Will remove: {samples_to_remove} samples (limited by constraints)")
        
        if samples_to_remove <= 0:
            logging.info(f"  No samples will be removed for {col}")
            continue
        
        # Remove samples from the majority class
        if minority_class == "positive":
            # Remove negative samples
            majority_indices = balanced_df[balanced_df[col] == 0].index.tolist()
        else:
            # Remove positive samples  
            majority_indices = balanced_df[balanced_df[col] == 1].index.tolist()
        
        # Don't remove more than available
        samples_to_remove = min(samples_to_remove, len(majority_indices))
        
        if samples_to_remove > 0:
            # Randomly select samples to remove
            remove_indices = np.random.choice(majority_indices, size=samples_to_remove, replace=False)
            balanced_df = balanced_df.drop(index=remove_indices)
            balanced_df = balanced_df.reset_index(drop=True)
            total_removed += len(remove_indices)
            
            # Log the effect
            new_minority_count = balanced_df[col].sum() if minority_class == "positive" else (len(balanced_df) - balanced_df[col].sum())
            new_total = len(balanced_df)
            new_minority_ratio = new_minority_count / new_total if new_total > 0 else 0
            
            logging.info(f"  Removed {len(remove_indices)} majority samples")
            logging.info(f"  New ratio: {new_minority_ratio:.3f} (target: {target_min_ratio:.3f})")
            
            # Check if we've reached our removal budget
            if total_removed >= max_allowed_removal:
                logging.info(f"  Reached maximum removal budget ({max_allowed_removal} samples) - stopping balancing")
                break
    
    # Final statistics
    logging.info(f"\nFinal {split_name} class distribution after balancing:")
    all_adequately_balanced = True
    
    for col in label_columns:
        positive = balanced_df[col].sum()
        total = len(balanced_df)
        negative = total - positive
        positive_ratio = positive / total if total > 0 else 0
        negative_ratio = negative / total if total > 0 else 0
        
        minority_ratio = min(positive_ratio, negative_ratio)
        minority_class = "positive" if positive_ratio < negative_ratio else "negative"
        
        is_adequate = minority_ratio >= target_min_ratio
        status = "✓" if is_adequate else "✗"
        
        if not is_adequate:
            all_adequately_balanced = False
            
        logging.info(f"  {col}: {positive} pos ({positive_ratio:.3f}), {negative} neg ({negative_ratio:.3f}) {status}")
        logging.info(f"    Minority: {minority_class} ({minority_ratio:.3f})")
        
        if minority_ratio < 0.1:
            logging.warning(f"    {col} still very imbalanced (minority: {minority_ratio:.3f})")
        elif minority_ratio < target_min_ratio:
            logging.warning(f"    {col} below target but improved (minority: {minority_ratio:.3f})")
    
    if all_adequately_balanced:
        logging.info(f"✓ All classes in {split_name} now have adequate representation (>= {target_min_ratio:.3f})")
    else:
        logging.warning(f"✗ Some classes in {split_name} still below target, but improved")
    
    reduction_percentage = (total_removed / original_size) * 100
    logging.info(f"{split_name} dataset size: {original_size} -> {len(balanced_df)} (removed {total_removed} samples, {reduction_percentage:.1f}% reduction)")
    
    # Final safety check
    if len(balanced_df) == 0:
        logging.error(f"ERROR: {split_name} dataset was completely removed during balancing! Returning original dataset.")
        return df
    
    return balanced_df

def skip_balancing_and_warn(df, split_name):
    """
    Skip balancing but provide detailed analysis of class imbalance.
    Recommend using class weights instead.
    """
    logging.info(f"Skipping dataset balancing for {split_name} - will use class weights instead")
    
    label_columns = ['adult_person_present', 'child_person_present', 'adult_face_present', 
                    'child_face_present', 'object_interaction']
    
    logging.info(f"\n{split_name} class distribution analysis:")
    extremely_imbalanced = []
    
    for col in label_columns:
        positive = df[col].sum()
        total = len(df)
        negative = total - positive
        positive_ratio = positive / total if total > 0 else 0
        negative_ratio = negative / total if total > 0 else 0
        
        minority_ratio = min(positive_ratio, negative_ratio)
        minority_class = "positive" if positive_ratio < negative_ratio else "negative"
        majority_ratio = max(positive_ratio, negative_ratio)
        
        # Calculate imbalance ratio
        imbalance_ratio = majority_ratio / minority_ratio if minority_ratio > 0 else float('inf')
        
        logging.info(f"  {col}:")
        logging.info(f"    Positive: {positive}/{total} ({positive_ratio:.3f})")
        logging.info(f"    Negative: {negative}/{total} ({negative_ratio:.3f})")
        logging.info(f"    Imbalance ratio: {imbalance_ratio:.1f}:1 ({minority_class} is minority)")
        
        if minority_ratio < 0.05:  # Less than 5%
            extremely_imbalanced.append(col)
            logging.warning(f"    ⚠️  EXTREMELY imbalanced! Minority class < 5%")
        elif minority_ratio < 0.15:  # Less than 15%
            logging.warning(f"    ⚠️  Severely imbalanced! Minority class < 15%")
        elif minority_ratio < 0.3:  # Less than 30%
            logging.info(f"    ℹ️  Moderately imbalanced")
        else:
            logging.info(f"    ✓ Reasonably balanced")
    
    if extremely_imbalanced:
        logging.warning(f"\n⚠️  Classes with extreme imbalance (< 5% minority): {extremely_imbalanced}")
        logging.warning("   These classes may be difficult to learn. Consider:")
        logging.warning("   1. Using very high class weights")
        logging.warning("   2. Focal loss instead of BCE")
        logging.warning("   3. Collecting more data for minority class")
        logging.warning("   4. Using specialized sampling techniques")
    
    logging.info(f"\nRecommendation: Use class weights in loss function instead of dataset balancing")
    logging.info("Class weights will be calculated automatically based on these distributions")
    
    return df
       
if __name__ == "__main__":
    logging.info("=== Starting CNN annotations extraction ===")
    
    RAW_OUTPUT_PATH = "/home/nele_pauline_suffo/ProcessedData/cnn_labels/cnn_annotations_raw.csv"
    MULTILABEL_OUTPUT_PATH = "/home/nele_pauline_suffo/ProcessedData/cnn_labels/cnn_annotations_multilabel.csv"
    AUGMENTED_OUTPUT_PATH = "/home/nele_pauline_suffo/ProcessedData/cnn_labels/cnn_annotations_augmented.csv"   
    SPLITS_OUTPUT_DIR = "/home/nele_pauline_suffo/ProcessedData/cnn_input"
    try:
        # # Fetch annotations for person and object categories
        # logging.info("Step 1: Fetching annotations from database")
        # annotations = fetch_all_annotations(CATEGORY_IDS)

        # # Convert to DataFrame
        # logging.info("Step 2: Converting annotations to DataFrame")
        # df_annotations = pd.DataFrame(annotations, columns=['category_id', 'bbox', 'object_interaction', 'file_name', 'gaze_directed_at_child', 'person_age'])
        # logging.info(f"DataFrame created with shape: {df_annotations.shape}")

        # # Save raw annotations to CSV
        # logging.info(f"Step 3: Saving raw annotations to CSV: {RAW_OUTPUT_PATH}")
        # df_annotations.to_csv(RAW_OUTPUT_PATH, index=False)
        # logging.info(f"Successfully saved {len(df_annotations)} raw annotations to CSV")
        
        # # Convert to multi-label format
        # logging.info("Step 4: Converting to multi-label format")
        # multilabel_df = convert_to_multilabel_format(RAW_OUTPUT_PATH, MULTILABEL_OUTPUT_PATH)
        
        # # Log some basic statistics
        # logging.info("=== Dataset Statistics ===")
        # logging.info(f"Total raw annotations: {len(df_annotations)}")
        # logging.info(f"Total frames after conversion: {len(multilabel_df)}")
        # logging.info(f"Unique frames in raw data: {df_annotations['file_name'].nunique()}")
        
        # # Category distribution in raw data
        # category_dist = df_annotations['category_id'].value_counts().sort_index()
        # logging.info("Category distribution in raw dataset:")
        # for cat_id, count in category_dist.items():
        #     logging.info(f"  Category {cat_id}: {count} annotations")
            
        # Randomly sample frames without detections
        #logging.info("Step 5: Augmenting with random frames")
        #augment_with_random_frames(MULTILABEL_OUTPUT_PATH, AUGMENTED_OUTPUT_PATH)
        
        # Create stratified splits
        logging.info("Step 6: Creating stratified splits by child")
        train_df, val_df, test_df, child_assignment = create_stratified_splits_by_child(
            AUGMENTED_OUTPUT_PATH, 
            SPLITS_OUTPUT_DIR,
            test_size=0.15,    # 10% test
            val_size=0.15,     # 10% val  
            random_state=42   # This should give ~80% train
        )
        
        # Balance train and validation sets
        logging.info("Step 7: Balancing class distributions in train and validation sets")
        
        # Balance training set with more conservative parameters
        train_df_balanced = balance_dataset_classes(
            train_df, 
            "Train", 
            target_min_ratio=0.25,  # 25% minimum representation (more conservative)
            max_reduction=0.4,      # Allow max 40% reduction
            random_state=42
        )
        
        # Balance validation set with even more conservative parameters
        val_df_balanced = balance_dataset_classes(
            val_df, 
            "Validation", 
            target_min_ratio=0.20,  # 20% minimum representation (more conservative for smaller val set)
            max_reduction=0.3,      # Allow max 30% reduction
            random_state=43        )
        
        # Keep test set unbalanced for realistic evaluation
        logging.info("Keeping test set unbalanced for realistic evaluation")
        
        # Save the balanced datasets
        train_path_balanced = os.path.join(SPLITS_OUTPUT_DIR, 'train_annotations_balanced.csv')
        val_path_balanced = os.path.join(SPLITS_OUTPUT_DIR, 'val_annotations_balanced.csv')
        test_path_original = os.path.join(SPLITS_OUTPUT_DIR, 'test_annotations.csv')  # Keep original test set
        
        train_df_balanced.to_csv(train_path_balanced, index=False)
        val_df_balanced.to_csv(val_path_balanced, index=False)
        # test_df is already saved in the original location
        
        logging.info(f"Saved balanced datasets:")
        logging.info(f"  Balanced Train: {train_path_balanced}")
        logging.info(f"  Balanced Val: {val_path_balanced}")
        logging.info(f"  Original Test: {test_path_original}")
        
        # Update child assignment with balanced statistics
        def calculate_balanced_split_stats(split_df, split_name):
            """Calculate updated statistics for balanced split."""
            label_columns = ['adult_person_present', 'child_person_present', 'adult_face_present', 
                            'child_face_present', 'object_interaction']
            stats = {}
            total_samples = len(split_df)
            
            for col in label_columns:
                positive_count = split_df[col].sum()
                negative_count = total_samples - positive_count
                positive_ratio = positive_count / total_samples if total_samples > 0 else 0
                negative_ratio = negative_count / total_samples if total_samples > 0 else 0
                
                stats[col] = {
                    'positive_count': int(positive_count),
                    'negative_count': int(negative_count),
                    'total_count': int(total_samples),
                    'positive_ratio': float(positive_ratio),
                    'negative_ratio': float(negative_ratio),
                    'positive_percentage': float(positive_ratio * 100),
                    'negative_percentage': float(negative_ratio * 100)
                }
            
            return stats
        
        # Calculate balanced statistics
        train_balanced_stats = calculate_balanced_split_stats(train_df_balanced, "Train_Balanced")
        val_balanced_stats = calculate_balanced_split_stats(val_df_balanced, "Val_Balanced")
        
        # Add balanced statistics to child assignment
        child_assignment['balanced_datasets'] = {
            'train_balanced_frames': len(train_df_balanced),
            'val_balanced_frames': len(val_df_balanced),
            'train_original_frames': len(train_df),
            'val_original_frames': len(val_df),
            'train_removed_frames': len(train_df) - len(train_df_balanced),
            'val_removed_frames': len(val_df) - len(val_df_balanced),
            'class_distribution_balanced': {
                'train': train_balanced_stats,
                'val': val_balanced_stats,
                'test': child_assignment['class_distribution']['test']  # Keep original test stats
            },
            'balancing_info': {
                'target_ratio_range': [0.3, 0.7],  # 30% to 70% for balanced splits
                'train_random_state': 42,
                'val_random_state': 43,
                'test_kept_original': True,
                'minimum_minority_class_ratio': 0.2,  # Updated to 20%
                'note': 'Balanced to ensure at least 20% representation for minority class (80/20 split)'
            }
        }
        
        # Save updated child assignment
        assignment_path = os.path.join(SPLITS_OUTPUT_DIR, 'child_split_assignment.json')
        import json
        with open(assignment_path, 'w') as f:
            json.dump(child_assignment, f, indent=2)
        logging.info(f"Updated child assignment saved to: {assignment_path}")
        
        # Log final summary
        logging.info("\n=== Final Dataset Summary ===")
        logging.info("Original datasets:")
        logging.info(f"  Train: {len(train_df)} samples")
        logging.info(f"  Val: {len(val_df)} samples")
        logging.info(f"  Test: {len(test_df)} samples")
        
        logging.info("Balanced datasets:")
        logging.info(f"  Train: {len(train_df_balanced)} samples (removed {len(train_df) - len(train_df_balanced)})")
        logging.info(f"  Val: {len(val_df_balanced)} samples (removed {len(val_df) - len(val_df_balanced)})")
        logging.info(f"  Test: {len(test_df)} samples (kept original)")
        
        total_original = len(train_df) + len(val_df) + len(test_df)
        total_balanced = len(train_df_balanced) + len(val_df_balanced) + len(test_df)
        
        logging.info(f"Total samples: {total_original} -> {total_balanced} (removed {total_original - total_balanced})")
        
        # Log class balance comparison
        label_columns = ['adult_person_present', 'child_person_present', 'adult_face_present', 
                        'child_face_present', 'object_interaction']
        
        logging.info("\n=== Class Balance Comparison ===")
        for col in label_columns:
            logging.info(f"\n{col}:")
            
            # Original train
            orig_train_pos = train_df[col].sum()
            orig_train_total = len(train_df)
            orig_train_ratio = orig_train_pos / orig_train_total if orig_train_total > 0 else 0
            
            # Balanced train
            bal_train_pos = train_df_balanced[col].sum()
            bal_train_total = len(train_df_balanced)
            bal_train_ratio = bal_train_pos / bal_train_total if bal_train_total > 0 else 0
            
            # Original val
            orig_val_pos = val_df[col].sum()
            orig_val_total = len(val_df)
            orig_val_ratio = orig_val_pos / orig_val_total if orig_val_total > 0 else 0
            
            # Balanced val
            bal_val_pos = val_df_balanced[col].sum()
            bal_val_total = len(val_df_balanced)
            bal_val_ratio = bal_val_pos / bal_val_total if bal_val_total > 0 else 0
            
            logging.info(f"  Train: {orig_train_ratio:.3f} -> {bal_train_ratio:.3f}")
            logging.info(f"  Val:   {orig_val_ratio:.3f} -> {bal_val_ratio:.3f}")
            
        logging.info("=== CNN annotations extraction completed successfully ===")
        
    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}")
        raise
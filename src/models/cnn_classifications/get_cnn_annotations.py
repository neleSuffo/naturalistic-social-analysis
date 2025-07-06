import sqlite3
import pandas as pd
import re
import random
import logging
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

            # Sample the same number of random frames (if enough available)
            n_sample = min(n_annotated, len(valid_frames))
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
    
if __name__ == "__main__":
    logging.info("=== Starting CNN annotations extraction ===")
    
    RAW_OUTPUT_PATH = "/home/nele_pauline_suffo/ProcessedData/quantex_annotations/cnn_annotations_raw.csv"
    MULTILABEL_OUTPUT_PATH = "/home/nele_pauline_suffo/ProcessedData/quantex_annotations/cnn_annotations_multilabel.csv"
    AUGMENTED_OUTPUT_PATH = "/home/nele_pauline_suffo/ProcessedData/quantex_annotations/cnn_annotations_augmented.csv"
    
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
            
        # randomly sample frames without detections
        # logging.info("Step 5: Augmenting with random frames")
        # augment_with_random_frames(MULTILABEL_OUTPUT_PATH, AUGMENTED_OUTPUT_PATH)
        
        logging.info("=== CNN annotations extraction completed successfully ===")
        
    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}")
        raise
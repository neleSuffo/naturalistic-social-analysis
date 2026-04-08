import cv2
import pandas as pd
import subprocess
import sys
import argparse
from pathlib import Path

def create_annotated_video(video_path, segments_csv_path, frame_csv_path, output_dir):
    video_name = Path(video_path).stem
    print(f"🛠️ Initializing Corrected Pipeline for: {video_name}")

    # 1. Load Data
    try:
        segments_df = pd.read_csv(segments_csv_path)
        frame_data = pd.read_csv(frame_csv_path)
        segments_df.columns = segments_df.columns.str.strip()
        frame_data.columns = frame_data.columns.str.strip()
    except Exception as e:
        print(f"❌ CSV Load Error: {e}")
        return

    # 2. Optimization: Filter for this video
    video_frame_data = frame_data[frame_data['video_name'] == video_name]
    # Frame lookup is still okay for rules because they are sparse/sampled
    detection_lookup = {int(row['frame_number']): row.to_dict() for _, row in video_frame_data.iterrows()}
    
    # 3. SEGMENT LOGIC: Create an array where the index is the frame number
    # This is the "Bullet-Proof" way to ensure every frame has a label
    video_segments = segments_df[segments_df['video_name'] == video_name]
    
    # Get total frames to initialize our label array
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Pre-fill a list with "No Data" and overwrite with segment labels
    # Use a NumPy array for speed if memory allows, or a list
    frame_labels = ["No Data"] * (total_frames + 1)
    for _, seg in video_segments.iterrows():
        start = int(seg['segment_start'])
        end = int(seg['segment_end'])
        label = str(seg['interaction_type'])
        # Fill the range
        for f in range(start, min(end + 1, total_frames + 1)):
            frame_labels[f] = label

    # 4. FFMPEG Setup
    final_output = Path(output_dir) / f"{video_name}_annotated.mp4"
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', '-', 
        '-i', str(video_path), 
        '-map', '0:v:0', '-map', '1:a:0?', 
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-shortest',
        str(final_output)
    ]
    
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=sys.stderr, bufsize=10**8)

    frame_num = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # 1. Get State from our pre-filled array (Continuous)
            state_label = frame_labels[frame_num] if frame_num < len(frame_labels) else "End of Data"
            
            # 2. Get Rules from frame_level data (Sampled - Aligned to nearest 10)
            # Your analysis is every 10 frames, so we check the 'grid' frame
            aligned_frame = (frame_num // 10) * 10 
            curr_det = detection_lookup.get(aligned_frame, {})

            # --- DRAWING ---
            cv2.rectangle(frame, (0, height-140), (width, height), (0, 0, 0), -1)
            
            # Color coding for state
            color = (255, 255, 255) # White default
            if state_label == 'Interacting': color = (0, 255, 0) # Green
            elif state_label == 'Available': color = (0, 255, 255) # Yellow
            elif state_label == 'Alone': color = (0, 0, 255) # Red

            cv2.putText(frame, f"STATE: {state_label}", (30, height-100), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)

            # Rules display (Sampled data)
            rules = []
            if curr_det.get('rule1_turn_taking'): rules.append("TT")
            if curr_det.get('rule2_close_proximity'): rules.append("Prox")
            if curr_det.get('rule3_kcds_speaking'): rules.append("CDS")
            
            rule_text = f"Rules (sampled): {', '.join(rules) if rules else 'None'}"
            cv2.putText(frame, rule_text, (30, height-50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            process.stdin.write(frame.tobytes())
            
            frame_num += 1
            if frame_num % 1000 == 0:
                print(f"⏳ {video_name}: {(frame_num/total_frames)*100:.1f}%")

    finally:
        cap.release()
        if process.stdin: process.stdin.close()
        process.wait()

    print(f"✅ Annotated: {final_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    args = parser.parse_args()
    
    out = Path(args.output_folder)
    create_annotated_video(
        args.video_path, 
        out / "interaction_segments.csv", 
        out / "frame_level_social_interactions.csv", 
        out
    )
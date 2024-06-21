import cv2
import os


def extract_frames_from_single_video(
    video_file: str, 
    output_dir: str, 
    fps: int
) -> None:
    """
    This function extracts frames from a single video file and saves them as images in the output directory.

    Parameters
    ----------
    video_file : str
        the path to the video file
    output_dir : str
        the directory to save the extracted frames
    fps : int
        the frames per second to extract
    """
    # Open the video file
    cap = cv2.VideoCapture(video_file)
    # Get the frame rate and calculate the frame interval
    frame_rate = round(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = int(frame_rate / fps)
    nr_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame in range(0, nr_frames, frame_interval):
        frame_id = frame // frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, image = cap.read()
        if success:
            # Save the frame as an image
            video_file_name = os.path.splitext(os.path.basename(video_file))[0]
            cv2.imwrite(os.path.join(output_dir, f"{video_file_name}_{frame_id:06d}.jpg"), image)
from src.projects.social_interactions.common.constants import VTCParameters
from moviepy.editor import VideoFileClip
from pathlib import Path
import tempfile
import logging
import cv2
import subprocess

# Function to extract frames from a video
def extract_frames(
    video_path: str, 
    output_dir: str
) -> list:
    """
    This function extracts frames from a video file and saves them to a directory.
    It then returns the list of extracted frames.

    Parameters
    ----------
    video_path : str
        the path to the video file
    output_dir : str
        the directory to save the extracted frames

    Returns
    -------
    list
        the list of extracted frame paths
    """
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    # Loop through frames
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frame_name = f'{video_path.stem}_frame_{i:06d}.jpg'
        frame_path = Path(output_dir) / frame_name        # Save frame
        cv2.imwrite(frame_path, frame)
        # Append frame path to list
        frames.append(frame_path)

    cap.release()
    return frames



def create_video_writer(
    output_path: str,
    frames_per_second: int,
    frame_width: int,
    frame_height: int,
) -> cv2.VideoWriter:
    """
    This function creates a VideoWriter object to write the output video.

    Parameters
    ----------
    output_path : str
        the path to the output video file
    frames_per_second : int
        the frames per second of the video
    frame_width : int
        the width of the frame
    frame_height : int
        the height of the frame

    Returns
    -------
    cv2.VideoWriter
        the video writer object
    """

    # Create a VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path, fourcc, frames_per_second, (frame_width, frame_height)
    )

    return out


def create_video_to_id_mapping(video_names: list) -> dict:
    """
    This function creates a dictionary with mappings from video names to ids.

    Parameters
    ----------
    video_names : list
        a list of video names without the file extension

    Returns
    -------
    dict
        a dictionary with mappings from video names to ids
    """
    # Create a dictionary with mappings from video names to ids
    # the first video has id 0, the second video has id 1, and so on
    video_id_dict = {video_name: i for i, video_name in enumerate(video_names)}
    return video_id_dict


def extract_audio_from_video(video: VideoFileClip, filename: str) -> None:
    """
    This function extracts the audio from a video file
    and saves it as a 16kHz WAV file.

    Parameters
    ----------
    video : VideoFileClip
        the video file
    filename : str
        the filename of the video
    """
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=True)

    # Extract the audio and save it to the temporary file
    video.audio.write_audiofile(temp_file.name + ".wav", codec="pcm_s16le")

    # Create the output directory if it doesn't exist
    VTCParameters.audio_path.mkdir(parents=True, exist_ok=True)

    # Convert the audio to 16kHz with sox and
    # save it to the output file
    output_file = VTCParameters.audio_path / f"{filename}{VTCParameters.audio_name_ending}"

    subprocess.run(
        ["sox", temp_file.name + ".wav", "-r", "16000", output_file],
        check=True,
    )
    
    # Delete the temporary file
    temp_file.close()
    
    logging.info(f"Successfully stored the file at {output_file}")


def extract_audio_from_videos_in_folder(folder_path: Path) -> None:
    """
    Extracts audio from all video files in the specified folder, if not already done.
    """
    for video_file in folder_path.iterdir():
        if video_file.suffix.lower() not in ['.mp4']:
            continue  # Skip non-video files
        
        audio_path = VTCParameters.audio_path / f"{video_file.stem}{VTCParameters.audio_name_ending}"
        
        # Check if the audio file already exists
        if not audio_path.exists():
            # Create a VideoFileClip object
            video_clip = VideoFileClip(str(video_file))  
            # Extract audio from the video
            extract_audio_from_video(video_clip, video_file.stem)  
            print(f"Extracted audio from {video_file.name}")
        else:
            print(f"Audio already exists for {video_file.name}")

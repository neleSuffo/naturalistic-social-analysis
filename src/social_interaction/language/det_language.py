import sys
import os
import numpy as np
import librosa
import cv2
import tempfile
from pydub import AudioSegment
from PIL import Image
from moviepy.editor import concatenate_videoclips, AudioFileClip, VideoFileClip
import speech_recognition as sr


# Get the grandparent directory of the current file
grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# Add the grandparent directory to the system path
sys.path.append(grandparent_dir)
# Import the utils module
from social_interaction import utils

def language_detection(video_input_path: str, 
                       video_output_path: str,
                       segment_duration: int,
                       bar_height: int=20) -> tuple[np.ndarray, list]:
    """
    This function loads a video from a given path, extracts the audio, and checks if spoken language is audible.
    It creates a detection bar where a green marker is added if speech is audible.

    Parameters
    ----------
    video_input_path : str
        the path to the video file
    video_output_path : str
        the path to the output video file
    segment_duration : int, optional
        the duration of each segment in seconds
    bar_height : int, optional
        the height of the bar, by default 20

    Returns
    -------
    np.ndarray
        the detection bar with green markers if speech is audible
    list
        the results for each segment (1 if speech is audible, 0 otherwise)
    """
    
    # Load video file and extract properties
    cap, frame_width, frame_height, frame_count, frames_per_second = utils.get_video_properties(video_input_path)
    # Create a VideoWriter object to write the video
    out = utils.create_video_writer(video_output_path, frames_per_second, frame_width, frame_height, 1)     
    
    # Extract audio from video
    audio_file, audio_file_path = extract_audio(video_input_path)

    # Check if spoken language is audible in the audio file
    #detection_list = is_spoken_language_audible_lib(audio_file_path, segment_duration)

    detection_list = is_spoken_language_audible_sr(audio_file_path, segment_duration)
    detection_bar = frame_wise_language_detection_with_bar(cap, 
                                                           detection_list,
                                                           frame_width, 
                                                           frame_count, 
                                                           out,
                                                           bar_height)
    
    # Load the output video
    video_clip = VideoFileClip(video_output_path)

    # Set the audio of the output video to the extracted audio
    video_clip = video_clip.set_audio(AudioFileClip(audio_file_path))
    
    # Write the output video file with audio
    video_clip.write_videofile(video_output_path, codec='libx264', audio_codec='aac')
    return detection_bar, detection_list

    


def frame_wise_language_detection_with_bar(cap: cv2.VideoCapture, 
                                           detection_results_list: list,
                                           frame_width: int, 
                                           frame_count: int, 
                                           out: cv2.VideoWriter, 
                                           bar_height: int) -> np.ndarray:
    """
    This function performs frame-wise language detection and adds a detection bar to the bottom of the frame.
    If speech is audible in a segment, a green marker is added to the detection bar.

    Parameters
    ----------
    cap : cv2.VideoCapture
        the video capture object
    detection_results_list : list
        the results for each segment (1 if speech is audible, 0 otherwise)
    frame_width : int
        the width of the frame
    frame_count : int
        the total number of frames in the video
    out : cv2.VideoWriter
        the video writer object
    bar_height : int
        the height of the bar

    Returns
    -------
    np.ndarray
        the detection bar with green markers if speech is audible
    """
    # Create a detection bar equivalent to the length of the file
    detection_bar = np.full((bar_height, frame_width, 3), 128, dtype=np.uint8)
                                           
    # Loop through frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break    

        # Convert frame to PIL Image
        img = Image.fromarray(frame[..., ::-1])
        
        # Loop through detection results
        for i in detection_results_list:
            # Add a green marker to the detection bar if speech is audible
            if i == 1:
                # Calculate the marker position based on the current index and total length of the list
                marker_position = int((cap.get(cv2.CAP_PROP_POS_FRAMES) / frame_count) * frame_width)
                # Add a marker to the detection bar
                cv2.line(detection_bar, (marker_position, 0), (marker_position, bar_height), (128,134,123), thickness=2)
    
        # Write the frame with the detection bar to the output video
        out.write(np.vstack((frame, detection_bar)))
        
        # Display modified frame
        # Only for testing purposes
        # cv2.imshow('Object Detection', np.vstack((frame, detection_bar)))
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release video capture and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return detection_bar


def extract_audio(video_path: str) -> tuple[AudioSegment, str]:
    """
    This function extracts the audio from a video file and returns it as an MP3 object.

    Parameters
    ----------
    video_path : str
        the path to the video file

    Returns
    -------
    AudioSegment
        the extracted audio as an MP3 object
    str
        the path to the temporary audio file
    """
    # Load the video file
    video_clip = VideoFileClip(video_path)

    # Extract the audio and save it as a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    video_clip.audio.write_audiofile(temp_file.name, codec='libmp3lame')

    # Extract the audio and save it to the specified path
    #path = 'runs/test_audio.mp3'
    #video_clip.audio.write_audiofile(path, codec='libmp3lame')

    # Load the temporary file as an MP3 object
    audio_file = AudioSegment.from_mp3(temp_file.name)
    #audio_file = AudioSegment.from_mp3(path)

    # Return the MP3 object and the path to the temporary file
    return audio_file, temp_file.name

# This function checks if spoken language is audible in an audio file
def is_spoken_language_audible_lib(audio_file: str,
                                   segment_duration: int) -> list:
    """
    This function checks if spoken language is audible in an audio file. 
    It extracts segments of the audio file and calculates the short-term energy of each segment. 
    If the average energy of a segment exceeds a threshold, the segment is considered to contain audible speech.

    Parameters
    ----------
    audio_file : str
        the path to the audio file
    segment_duration : int, optional
        the duration of each segment in seconds
    
    Returns
    -------
    list
        the results for each segment (1 if speech is audible, 0 otherwise)
    """
    # Load audio file
    y, sr = librosa.load(audio_file)

    # Calculate the duration of the audio in seconds
    duration = librosa.get_duration(y=y, sr=sr)

    # Initialize a list to store the results for each segment
    detection_list = []

    # Iterate over each segment
    for i in range(0, int(duration), segment_duration):
        # Extract the segment
        segment = y[i * sr : (i + segment_duration) * sr]

        # Calculate the short-term energy of the segment
        energy = librosa.feature.rms(y=segment)

        # Calculate the average energy
        avg_energy = energy.mean()

        # Set a threshold for average energy to determine if speech is audible
        # Adjust as needed
        threshold = 0.015  

        # Check if average energy exceeds the threshold
        if avg_energy > threshold:
            detection_list.append(1)
        else:
            detection_list.append(0)

    return detection_list

def is_spoken_language_audible_sr(audio_path: str,
                                  segment_duration) -> list:
    """
    This function checks if spoken language is audible in an audio file using the SpeechRecognition library.
    If speech is detected in a segment, the function appends 1 to the detection list; otherwise, it appends 0.

    Parameters
    ----------
    audio_path : str
        the path to the audio file
    segment_duration : int, optional
        the size of the audio segments in seconds

    Returns
    -------
    list
        the results for each segment (1 if speech is audible, 0 otherwise)
    """
    
    # Initialize speech recognizer
    recognizer = sr.Recognizer()

    # Initialize list to store speech detection results
    detection_list = []

    # Load mp3 file
    audio = AudioSegment.from_mp3(audio_path)

    # Convert to wav
    wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
    audio.export(wav_path, format="wav")

    # Calculate total duration in seconds
    total_duration = len(audio) / 1000

    # Open audio file
    with sr.AudioFile(wav_path) as source:
        
        # Iterate over each segment
        for start_time in range(0, int(total_duration), segment_duration):
            # Load audio data segment
            audio_data_segment = recognizer.record(source, duration=segment_duration, offset=start_time * segment_duration)

            # Use recognizer to recognize speech in the segment
            try:
                recognizer.recognize_google(audio_data_segment)
                # If speech is detected, append 1 to the detection list
                detection_list.append(1)  
            except sr.UnknownValueError:
                # If no speech is detected, append 0 to the detection list
                detection_list.append(0)

    return detection_list



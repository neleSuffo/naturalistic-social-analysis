from persons import det_persons
from language import det_language
from faces import dlib, MTCNN
from moviepy.editor import VideoFileClip

video_input_path = "data/sample_2_short.mp4"
video_output_1_path = "runs/sample_2_det_persons.mp4"
video_output_2_path = "runs/sample_2_det_language.mp4"

def run_combined_detection(video_input_path: str, 
                           video_output_1_path: str, 
                           video_output_2_path: str,
                           audio_detection_segment_duration: int=1):
    """This function runs the combined detection of persons and spoken language in a video file.
    It then prints the percentages of person visible relative to the total frames and spoken language relative to the length of the audio file.

    Parameters
    ----------
    video_input_path : str
        the path to the video file
    video_output_1_path : str
        the path to the output video file for person detection  
    video_output_2_path : str
        the path to the output video file for language detection
    audio_detection_segment_duration : int, optional
        the duration of the audio segments for language detection, by default 1
    """

    # Perform person detection on the video
    #persons_detection_bar, person_detection_list = det_persons.person_detection(video_input_path, video_output_1_path)

    # Perform language detection on the audio and create a grey bar with green markers
    #language_detection_bar, language_detection_list = det_language.language_detection(video_input_path, video_output_2_path, audio_detection_segment_duration)

    #dlib.detect_faces(video_input_path)
    dlib.detect_faces(video_input_path)


    # Print the percentages of person visible and spoken language
    #print(f'Percentages of person visbile relative to the total frames: {sum(person_detection_list) / len(person_detection_list)*100}')
    #print(f'Percentages of spoken language relative to the length of the audio file: {sum(language_detection_list) / len(language_detection_list)*100}')

    # Concatenate the output video containing the bounding boxes with the detection bars
    #utils.concat_video_with_bars(video_storage_path, video_output_path, persons_detection_bar, language_detection_bar)

#if __name__ == "__main__":
#    run_combined_detection(video_input_path, video_output_1_path, video_output_2_path)


# convert mp4 to wav
def extract_audio(video_input_path: str, audio_output_path: str):
    """Extracts the audio from a video file and exports it as a WAV file.

    Parameters
    ----------
    video_input_path : str
        the path to the video file
    audio_output_path : str
        the path to export the audio as a WAV file
    """
    video = VideoFileClip(video_input_path)
    audio = video.audio
    audio.write_audiofile(audio_output_path, codec='pcm_s16le')

# Example usage
video_input_path = "data/sample_2_short.MP4"
audio_output_path = "data/sample_2.wav"
extract_audio(video_input_path, audio_output_path)
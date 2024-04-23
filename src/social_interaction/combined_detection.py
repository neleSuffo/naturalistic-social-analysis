import os

from faces import MTCNN
from language import detect_voices

from config import video_face_output_path, videos_input_path


def run_social_interactions_detection(video_input_path: str) -> None:  # noqa: E501
    """
    This function runs the combined detection of persons
    and spoken language in a video file.
    It then prints the percentages of person visible relative
    to the total frames and spoken language
    relative to the length of the audio file.

    Parameters
    ----------
    video_input_path : str
        the path to the video file
    """
    # PERSONS
    # Perform person detection on the video
    # person_detection_list = det_persons.person_detection(video_input_path, video_person_output_path)  # noqa: E501
    # Get the percentage of person visible relative to the total frames
    # person_percentage = sum(person_detection_list) / len(person_detection_list)*100  # noqa: E501
    # print(f'Percentages of at least one person visible relative to the total frames: {person_percentage}')  # noqa: E501

    # FACES
    face_detection_list = MTCNN.face_detection(
        video_input_path, video_face_output_path
    )  # noqa: E501
    # Get the percentage of face visible relative to the total frames
    face_percentage = sum(face_detection_list) / len(face_detection_list) * 100
    print(
        f"Percentages of at least one face visible relative to the total frames: {face_percentage:.2f}"  # noqa: E501, E231
    )  # noqa: E501, E231
    print(
        f"Total number of frames: {len(face_detection_list)}, Number of frames with faces: {sum(face_detection_list)}"  # noqa: E501
    )  # noqa: E501, E231

    # VOICE
    (
        total_video_duration,
        voice_duration_sum,
    ) = detect_voices.extract_speech_duration(  # noqa: E501
        video_input_path
    )
    # Get the percentage of spoken language relative to the length of the audio file  # noqa: E501
    # TODO: Adjust percentage calculation when more than one file is processed
    voice_percentage = voice_duration_sum / total_video_duration * 100
    print(
        f"Percentages of spoken language relative to the length of the audio file: {voice_percentage:.2f}%"  # noqa: E501, E231
    )
    print(
        f"Total video duration: {total_video_duration:.2f}, Total voice duration: {voice_duration_sum:.2f}"  # noqa: E501, E231
    )  # noqa: E501, E231

    # PROXIMITY


if __name__ == "__main__":
    # Get a list of all video files in the folder
    video_files = [
        f
        for f in os.listdir(videos_input_path)
        if f.lower().endswith(".mp4")  # noqa: E501
    ]
    # Process each video file
    for video_file in video_files:
        video_path = os.path.join(videos_input_path, video_file)  # noqa: E501
        run_social_interactions_detection(video_path)

import cv2


def get_video_nr_frames(file_path: str) -> int:
    """
    This function returns the number of frames of a video file.

    Parameters
    ----------
    file_path : str
        the path to the video file

    Returns
    -------
    int
        the number of frames of the video
    """
    cap = cv2.VideoCapture(file_path)
    nr_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return nr_frames


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

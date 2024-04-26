from facenet_pytorch import MTCNN
import torch
import cv2
from imutils.video import FileVideoStream


device = "cuda" if torch.cuda.is_available() else "cpu"

filenames = "/Users/nelesuffo/projects/leuphana-IPE/data/sample_2.MP4"


class FastMTCNN(object):
    """
    Fast MTCNN implementation.
    """

    def __init__(self, stride, resize=1, *args, **kwargs):
        """
        Constructor for FastMTCNN class.

        Parameters
        ----------
        stride : int
            the detection stride. Faces will be detected every `stride` frames
            and remembered for `stride-1` frames.
        resize : float, optional
            fractional frame scaling. [default: {1}]
        *args
            positional arguments to pass to the MTCNN constructor (see help(MTCNN))
        **kwargs
            keyword arguments to pass to the MTCNN constructor (see help(MTCNN))
        """
        self.stride = stride
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)

    def __call__(self, frames):
        """
        Detect faces in frames using strided MTCNN.

        Arguments
        ---------
        frames : list
            a list of frames from a video

        Returns
        -------
        faces : list
            the faces detected in frames
        """
        if self.resize != 1:
            frames = [
                cv2.resize(
                    f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize))
                )
                for f in frames
            ]

        boxes, probs = self.mtcnn.detect(frames[:: self.stride])

        faces = []
        for i, frame in enumerate(frames):
            box_ind = int(i / self.stride)
            if boxes[box_ind] is None:
                continue
            for box in boxes[box_ind]:
                box = [int(b) for b in box]
                faces.append(frame[box[1] : box[3], box[0] : box[2]])

        return faces


fast_mtcnn = FastMTCNN(
    stride=4, resize=1, margin=14, factor=0.6, keep_all=True, device=device
)

# TODO: Complete the function
# run_detection(fast_mtcnn, filenames)


def run_face_detection(
    video_input_path: str, video_output_path: str, model: MTCNN
) -> list:
    """_summary_

    Parameters
    ----------
    video_input_path : str
        the path to the video file
    video_output_path : str
        the path to the output video file
    model : MTCNN
        the MTCNN face detector

    Returns
    -------
    list
        _description_
    """
    frames = []
    frames_processed = 0
    batch_size = 60

    # Load video and get video properties
    v_cap = FileVideoStream(video_input_path).start()
    v_len = int(v_cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))

    for j in range(v_len):
        # Load frame
        frame = v_cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

        # Perform face detection
        if len(frames) >= batch_size or j == v_len - 1:
            # faces = fast_mtcnn(frames)
            frames_processed += len(frames)
            frames = []

    v_cap.stop()

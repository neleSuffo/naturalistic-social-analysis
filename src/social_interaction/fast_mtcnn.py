from facenet_pytorch import MTCNN
import cv2


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
            the detection stride. Faces will be detected every 'stride' frames
            and remembered for 'stride-1' frames.
        resize : float, optional
            fractional frame scaling. [default: {1}]
        *args
            positional arguments to pass to the MTCNN constructor (see help(MTCNN))
        **kwargs
            keyword arguments to pass to the MTCNN constructor (see help(MTCNN))

        Returns
        -------

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
        detection_list : list
            the results for the batch of frames
            (1 if a face is detected, 0 otherwise)
        """
        # Resize frames
        if self.resize != 1:
            frames = [
                cv2.resize(
                    frame,
                    (
                        int(frame.shape[1] * self.resize),
                        int(frame.shape[0] * self.resize),
                    ),
                )
                for frame in frames
            ]

        # Detect faces for every 'stride' frames
        boxes, probs = self.mtcnn.detect(frames[:: self.stride])

        # Initialize detection list to store detection results
        detection_list = []
        # Iterate over frames
        for index, frame in enumerate(frames):
            # Detection results for N-frae are applied to all frames
            # e.g. batch = 9  and stride =3 : bounding boxes for frame 0
            # would be naively applied to frames 1 and 2 etc
            box_ind = int(index / self.stride)
            if boxes[box_ind] is None:
                # Append 0 to the detection list if no face is detected
                detection_list.append(0)
                continue
            # Append 1 to the detection list if at least one face is detected
            detection_list.append(1)

        return detection_list

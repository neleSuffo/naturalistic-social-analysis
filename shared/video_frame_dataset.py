import torch
import cv2
import json
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from projects.social_interactions.src.common.constants import DetectionPaths
from sklearn.model_selection import train_test_split
from shared import utils


class VideoFrameDataset(Dataset):
    """the VideoFrameDataset class is a custom 
    dataset class that loads video frames and
    their corresponding bounding boxes from a
    given list of annotations.

    Parameters
    ----------
    Dataset : 
        the dataset class
    """
    def __init__(self, annotations, transform=None):
        self.annotations = annotations
        self.transform = transform
        self.cap = None
        self.current_video_id = None


    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, idx: int) -> tuple:
        """
        This method returns the video frame and the bounding box

        Parameters
        ----------
        idx : int
            the index of the annotation

        Returns
        -------
        tuple
            the video frame, bounding box, and category id
        Raises
        ------
        ValueError
            the frame could not be read
        """
        annotation = self.annotations[idx]
        _, frame_id, video_id, category_id, bbox, _, video_file_name = annotation
        bbox = json.loads(bbox)  
        
        if self.cap is None or self.current_video_id != video_id:
            if self.cap is not None:
                self.cap.release()
            video_file_path = os.path.join(DetectionPaths.videos_input, video_file_name)
            self.cap = cv2.VideoCapture(video_file_path)
            self.current_video_id = video_id
            
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.cap.read()

        if not ret:
            raise ValueError(f"Could not read frame {frame_id} from {video_file_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            frame = self.transform(frame)

        bbox = torch.tensor(bbox, dtype=torch.float32)
        
        return frame, bbox, category_id
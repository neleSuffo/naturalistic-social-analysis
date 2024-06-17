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
    def __init__(self, annotations, transform=None):
        self.annotations = annotations
        self.transform = transform
        self.cap = None
        self.current_video_file_path = None

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx: int) -> tuple:
        annotation = self.annotations[idx]
        _, _, _, bbox, image_file_name, video_file_name = annotation
        bbox = json.loads(bbox)  
        
        video_file_path = os.path.join(DetectionPaths.videos_input, video_file_name)
        if self.cap is None or self.current_video_file_path != video_file_path:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(video_file_path)
            self.current_video_file_path = video_file_path
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(image_file_name.split('_')[-1].split('.')[0]))
        ret, frame = self.cap.read()

        if not ret:
            raise ValueError(f"Could not read frame from {video_file_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            frame = self.transform(frame)

        bbox = torch.tensor(bbox, dtype=torch.float32)
        
        return frame, bbox
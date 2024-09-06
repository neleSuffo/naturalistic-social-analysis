import torch
from fastreid.config import get_cfg
from fastreid.modeling import build_model
from fastreid.utils.checkpoint import Checkpointer
from fastreid.data.transforms import T
from pathlib import Path
from src.projects.social_interactions.common.constants import FastReIDParameters as FRP, StrongSortParameters as SSP
import cv2
import numpy as np
import logging


def setup_model(
    config_file: str, 
    model_weights: str
) -> torch.nn.Module:
    """
    Loads a pretrained Fast ReID model using the given configuration file and model weights.

    Parameters
    ----------
    config_file : str
        Path to the configuration file.
    model_weights : str
        Path to the model weights.

    Returns
    -------
    torch.nn.Module
        The loaded Fast ReID model ready for inference.
    """
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build and load model
    model = build_model(cfg)
    model.eval()
    Checkpointer(model).load(model_weights)

    return model


def preprocess_image(
    image: torch.Tensor
) -> torch.Tensor:
    """
    Applies transformations to the input image for Fast ReID model inference.

    Parameters
    ----------
    image : torch.Tensor
        The input image.

    Returns
    -------
    torch.Tensor
        The preprocessed image ready for inference.
    """
    transform = T.ResizeShortestEdge([256, 256], 512)
    image = transform.get_transform(image).apply_image(image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))  # HWC to CHW
    return image.unsqueeze(0)  # Add batch dimension


def extract_features(
    model: torch.nn.Module, 
    image_path: Path
) -> torch.Tensor:
    """
    Extracts features from a single image using the Fast ReID model.

    Parameters
    ----------
    model : torch.nn.Module
        The pretrained Fast ReID model.
    image_path : Path
        The path to the image file.

    Returns
    -------
    torch.Tensor
        The extracted feature vector for the image.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Image {image_path} not found!")
        return None

    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Move image to GPU if available
    if torch.cuda.is_available():
        preprocessed_image = preprocessed_image.cuda()

    # Extract feature using Fast ReID model
    with torch.no_grad():
        feature = model(preprocessed_image)
    
    return feature.cpu().numpy()


def save_features(features, frame_id, output_dir):
    """
    Saves extracted feature vectors for a specific frame in .npy format.

    Parameters
    ----------
    features : list or np.ndarray
        The feature vectors extracted from the frame.
    frame_id : str
        The identifier for the frame (e.g., 'frame_000001').
    output_dir : Path
        The directory where the features will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_file = output_dir / f"{frame_id}.npy"
    
    # Save the feature vectors in .npy format
    np.save(feature_file, features)
    
    
def find_video_folder(
    video_name: str, 
) -> Path:
    """
    Searches for the given video folder name in 'train' or 'val' directories.

    Parameters
    ----------
    video_name : str
        The folder name of the video (e.g., 'quantex_at_home_id258239_2020_08_23').

    Returns
    -------
    Path
        The path to the found video directory, or None if not found.
    """
    try:
        base_dir = SSP.video_input
        for split in ['train', 'val']:
            potential_dir = base_dir / split / video_name
            if potential_dir.exists():
                return potential_dir
    except FileNotFoundError:
        logging.error(f"Video folder {video_name} not found in {base_dir}.")
        return None


def process_video_frames(
    video_folder: Path, 
    model: torch.nn.Module
):
    """
    Processes all images in video folders to extract features.

    Parameters
    ----------
    video_folder : Path
        The root folder containing multiple video frame subfolders.
    model : torch.nn.Module
        The Fast ReID model to use for feature extraction.
    """
    for video_folder_name in video_folder.iterdir():
        if video_folder_name.is_dir():
            for img_file in video_folder_name.iterdir():
                if img_file.is_file() and img_file.suffix in ['.jpg', '.png']:
                    # Extract features for each image
                    features = extract_features(model, img_file)
                    save_features(features, img_file.stem, video_folder_name)

# Function 2: Process frames and extract/save features
def process_video_frames_and_save_features(
    video_folder: Path,
    model: torch.nn.Module
):
    """
    Processes the frames in the given video directory and saves extracted features.

    Parameters
    ----------
    video_folder : Path
        The root folder containing multiple video frame subfolders.
    model : torch.nn.Module
        The Fast ReID model to use for feature extraction.
    """
    # Create a 'features' directory inside the video directory
    features_dir = video_folder / SSP.feature_subfolder
    features_dir.mkdir(parents=True, exist_ok=True)

    # Process all frames in the img1 folder
    for video_folder_name in video_folder.iterdir():
        if video_folder_name.is_dir():
            for img_file in video_folder_name.iterdir():
                if img_file.is_file() and img_file.suffix in ['.jpg', '.png']:
                    # Extract features for each image
                    features = extract_features(model, img_file)
                if features is not None:
                    # Get the frame identifier (e.g., '000001')
                    frame_id = img_file.stem
                    # Save the features
                    save_features(features, frame_id, features_dir)
                
                
def main():
    # Set the configuration and model weights paths
    config_file = FRP.config_file
    model_weights = FRP.pretrained_model

    # Load the pretrained model
    model = setup_model(config_file, model_weights)

    # Process and save features for all video frames
    process_video_frames_and_save_features(FRP.base_folder, model)

if __name__ == "__main__":
    main()
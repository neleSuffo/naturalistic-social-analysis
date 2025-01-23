import os
import cv2
import logging
import numpy as np
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
from huggingface_hub import hf_hub_download
from constants import YoloPaths

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def load_model(model_path: Path: YoloPaths.face_trained_weights_path) -> YOLO:
    """Load YOLO model from path"""
    try:
        model = YOLO(str(model_path))
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise
    
def process_image(model: YOLO, image_path: Path) -> Tuple[np.ndarray, Detections]:
    """Process image with YOLO model"""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
            
        output = model(Image.open(image_path))
        results = Detections.from_ultralytics(output[0])
        logging.info(f"Detected {len(results.xyxy)} faces")
        return image, results
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise

def draw_detections(image: np.ndarray, results: Detections) -> np.ndarray:
    """Draw bounding boxes and labels on image"""
    annotated_image = image.copy()
    for bbox, conf in zip(results.xyxy, results.confidence):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Face: {conf:.2f}"
        cv2.putText(annotated_image, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return annotated_image


def main():
    parser = argparse.ArgumentParser(description='YOLO Face Detection')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    args = parser.parse_args()
    
    setup_logging()
    
    # Initialize paths
    output_dir = YoloPaths.face_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load model and process image
        model = load_model(YoloPaths.face_trained_weights_path)
        image, results = process_image(model, Path(args.image))
        
        # Draw detections and save
        annotated_image = draw_detections(image, results)
        output_path = output_dir / Path(args.image).name
        cv2.imwrite(str(output_path), annotated_image)
        logging.info(f"Annotated image saved to: {output_path}")
        
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
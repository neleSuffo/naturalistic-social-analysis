import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Load the trained model
def load_model(model_path, device):
    model = models.resnet18(pretrained=False)  # Base ResNet-18 architecture
    model.fc = nn.Linear(model.fc.in_features, 2)  # Adjust for binary classification (2 classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    logger.info("Model loaded successfully.")
    return model

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Preprocess a single image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Predict gaze for a single image
def predict_gaze(model, device, image_path):
    image_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    label_map = {0: "No Gaze", 1: "Gaze"}
    predicted_label = label_map[predicted.item()]
    logger.info(f"Predicted gaze label for {image_path}: {predicted_label}")
    return predicted_label

# Predict gaze for a batch of images
def predict_gaze_batch(model, device, image_paths):
    predictions = {}
    for image_path in image_paths:
        predictions[image_path] = predict_gaze(model, device, image_path)
    return predictions

# Detect faces in a frame (example using MTCNN or another face detector)
def detect_faces(frame):
    # Example: Replace with your MTCNN implementation or any other face detection method
    logger.info("Detecting faces...")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces  # List of bounding boxes [(x, y, w, h), ...]

# Process a video frame for gaze predictions
def process_frame(frame, model, device):
    face_bboxes = detect_faces(frame)
    gaze_predictions = []
    for bbox in face_bboxes:
        x, y, w, h = bbox
        face = frame[y:y+h, x:x+w]  # Crop the face
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))  # Convert to PIL Image
        gaze_prediction = predict_gaze(model, device, face_pil)
        gaze_predictions.append((bbox, gaze_prediction))
    return gaze_predictions

# Main function
def main():
    # Define paths and device
    model_path = "/home/nele_pauline_suffo/models/best_gaze_classification_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = load_model(model_path, device)

    # Single image inference example
    test_image_path = "/home/nele_pauline_suffo/ProcessedData/quantex_faces/quantex_at_home_id255944_2022_03_10_01_026250_face_0.jpg"
    single_prediction = predict_gaze(model, device, test_image_path)
    logger.info(f"Prediction for single image: {single_prediction}")

    # # Batch inference example
    # batch_image_paths = [
    #     "/path/to/test_image1.jpg",
    #     "/path/to/test_image2.jpg",
    #     "/path/to/test_image3.jpg",
    # ]
    # batch_predictions = predict_gaze_batch(model, device, batch_image_paths)
    # for path, prediction in batch_predictions.items():
    #     logger.info(f"Prediction for {path}: {prediction}")

    # # Example: Process a video
    # video_path = "/path/to/video.mp4"
    # cap = cv2.VideoCapture(video_path)
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     predictions = process_frame(frame, model, device)
    #     for bbox, pred in predictions:
    #         x, y, w, h = bbox
    #         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #         cv2.putText(frame, pred, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    #     cv2.imshow("Video", frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import torch
import cv2
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# load DEEPLABV3 model
model = torch.hub.load("pytorch/vision:v0.10.0", "deeplabv3_resnet50", pretrained=True)
# or any of these variants
model.eval()

video_input_path = "/Users/nelesuffo/projects/leuphana-IPE/data/video/sample_2.MP4"


preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# create a color palette, selecting a color for each class
palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# Load video file
cap = cv2.VideoCapture(video_input_path)

# Loop through frames
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        # create a normalized tensor from the frame
        input_tensor = preprocess(frame)

        # create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0)

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to("cuda")
            model.to("cuda")

        with torch.no_grad():
            output = model(input_batch)["out"][0]
        output_predictions = output.argmax(0)

        # plot the semantic segmentation predictions of 21 classes in each color
        r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(
            frame.shape[1::-1]
        )
        r.putpalette(colors)

        plt.imshow(r)
        plt.show()

    else:
        break

# Release the VideoCapture object
cap.release()

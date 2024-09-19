# leuphana-IPE
PhD research

This project involves the use of several technologies for different types of detection:

1. **Person Detection**: For person detection, we used [YOLOv8]([https://github.com/ultralytics/yolov5](https://docs.ultralytics.com)). YOLOv5 is a state-of-the-art, real-time object detection system.

2. **Face Detection**: We utilized the `facenet_pytorch` library for face detection, which is a PyTorch adaptation of Google's FaceNet model. FaceNet, a deep learning model for facial recognition, was first presented by Google researchers in a paper named “FaceNet: A Unified Embedding for Face Recognition and Clustering” by Schroff et al. You can find the code for the MTCNN implementation on [GitHub](https://github.com/timesler/facenet-pytorch).

3. **Voice Detection**: For voice detection, we used an existing voice type classifier. You can find the code for this classifier on [GitHub](https://github.com/MarvinLvn/voice-type-classifier/tree/new_model).

By combining these techniques, we aim to capture a holistic view of social interactions, considering not just visual cues, but also auditory signals.

# Mask Detection using TensorFlow and OpenCV

This project involves training a deep learning model to detect whether a person is wearing a mask or not in real-time using a live video stream. The model is trained using the TensorFlow framework and utilizes the MobileNetV2 architecture for feature extraction. The face detection is performed using a pre-trained face detection model.


## Project Structure

The project is organized into two main scripts:

1. **`train.py`**: This script is responsible for loading and preprocessing the dataset, training the mask detection model, and saving the trained model for later use.

2. **`video.py`**: This script uses the trained model to perform real-time mask detection on a live video stream. It integrates a pre-trained face detection model to identify faces in each frame.

## Getting Started

### Prerequisites

- Install the required Python packages:

  
  pip install tensorflow opencv-python imutils scikit-learn matplotlib
  
# Running Real-time Mask Detection
Run the training script:
```
python train.py
```

Run the video script:
```
python video.py
```
This script will start a video stream, detect faces, and classify whether each person is wearing a mask or not.

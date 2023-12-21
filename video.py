from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

# Function to detect and predict masks in a given frame
def detect_and_predict_mask(frame, faceNet, maskNet):
	(h, w) = frame.shape[:2]

	# Preprocess the frame for face detection
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# Set the input to the face detection model
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# Lists to store faces, locations, and predictions
	faces = []
	locs = []
	preds = []

	# Loop through the detected faces
	for i in range(0, detections.shape[2]):
		
		confidence = detections[0, 0, i, 2]

		# If confidence is above a threshold, process the face
		if confidence > 0.5:
			# Calculate bounding box coordinates
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Ensure bounding box coordinates are within the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# Extract face, preprocess, and append to lists
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# If faces are detected, make predictions
	if len(faces) > 0:

		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	return (locs, preds)

# Path to face detection model files
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"

# Load the face detection model
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the trained mask detection model
maskNet = load_model("mask_detector.model")

# Start the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

while True:

	# Read a frame from the video stream
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# Detect faces and predict masks
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# Loop through the detected faces and display results
	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# Determine label and color based on the prediction
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# Display label and bounding box on the frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# Display the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# Check for the 'q' key to exit the loop
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()

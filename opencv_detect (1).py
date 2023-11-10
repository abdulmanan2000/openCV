from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv

# Load the trained model
model = load_model('Observe_Accident.model')

# Open the webcam
webcam = cv2.VideoCapture(0)

classes = ['Normal', 'Injure']

# Loop through frames
while webcam.isOpened():

    # Read frame from webcam
    status, frame = webcam.read()

    # Apply face detection
    face, confidence = cv.detect_face(frame)

    # Loop through detected faces
    for idx, f in enumerate(face):

        # Get corner points of face rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # Calculate additional dimensions for larger crop
        width = endX - startX
        height = endY - startY
        expand_ratio = 0.2  # Adjust the ratio as needed

        # Calculate expanded bounding box coordinates
        expanded_startX = max(0, startX - int(width * expand_ratio))
        expanded_startY = max(0, startY - int(height * expand_ratio))
        expanded_endX = min(frame.shape[1], endX + int(width * expand_ratio))
        expanded_endY = min(frame.shape[0], endY + int(height * expand_ratio))

        # Draw rectangle over expanded region
        cv2.rectangle(frame, (expanded_startX, expanded_startY), (expanded_endX, expanded_endY), (0, 255, 0), 2)

        # Crop the expanded region
        expanded_crop = np.copy(frame[expanded_startY:expanded_endY, expanded_startX:expanded_endX])

        if (expanded_crop.shape[0]) < 10 or (expanded_crop.shape[1]) < 10:
            continue

        # Preprocess the image for model input
        expanded_crop = cv2.resize(expanded_crop, (96, 96))
        expanded_crop = expanded_crop.astype("float") / 255.0
        expanded_crop = img_to_array(expanded_crop)
        expanded_crop = np.expand_dims(expanded_crop, axis=0)

        # Apply classification on the expanded crop
        conf = model.predict(expanded_crop)[0]
        normal_conf = conf[0] * 100

        # Choose color based on Normal confidence
        color = (0, 255, 0)  # Default color is green (Normal)

        if normal_conf < 30:
            color = (0, 0, 255)  # Set color to red (Injured)

        label = f'Normal: {normal_conf:.2f}%'

        # Write label above expanded rectangle
        cv2.putText(frame, label, (expanded_startX, expanded_startY - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

    # Display output
    cv2.imshow("Accident detection", frame)

    # Press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()

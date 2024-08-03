import cv2
import numpy as np
from keras._tf_keras.keras.models import load_model

def preprocess_frame(frame, target_size=(37, 50), color_mode='grayscale'):
    if color_mode == 'grayscale':
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, target_size)
    if color_mode == 'grayscale':
        frame = frame[..., np.newaxis]  # Add channel dimension for grayscale images
    frame = frame / 255.0  # Normalize to [0, 1]
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

def predict_frame(model, frame, target_size=(37, 50), color_mode='grayscale'):
    frame = preprocess_frame(frame, target_size, color_mode)
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    prediction = model.predict(frame)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class[0]

# Load the trained model
model = load_model('model.keras')  # Replace with your model path

# Open a connection to the webcam
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame.")
        break

    # Predict the class of the frame
    predicted_class = predict_frame(model, frame)
    
    # Display the predicted class on the frame
    cv2.putText(frame, f"Class: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow('Webcam', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
